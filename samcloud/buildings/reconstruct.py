"""End-to-end, failure-tolerant building reconstruction."""

from __future__ import annotations

import logging
import time

import numpy as np
from shapely.geometry import Point, Polygon

from samcloud.buildings.extrude import extrude_lod1, extrude_lod13, extrude_lod22
from samcloud.buildings.footprint import detect_facade_planes, footprint_from_facades
from samcloud.buildings.labeling import LabelResult, label_cells
from samcloud.buildings.partition import Partition, partition_footprint, partition_from_raster, plane_lines
from samcloud.buildings.regularize import regularize_footprint, regularize_roof_planes, regularize_settings
from samcloud.buildings.roof_planes import detect_raster_roof_planes, detect_roof_planes
from samcloud.buildings.validate import validate_model
from samcloud.drone.contracts import BuildingModel, PointSet, ROOF_FLAT, ROOF_INCLINED, RoofPlane, load_drone_config


logger = logging.getLogger(__name__)


def _failed(building_id: int, reason: str, timings: dict | None = None) -> BuildingModel:
    return BuildingModel(building_id, "1.2", np.empty((0, 3)), np.empty((0, 3), dtype=int), np.empty(0, dtype=np.uint8), "failed", reason, heights={"timings": timings or {}})


def _ground_value(sampler, footprint: Polygon) -> float:
    point = footprint.representative_point()
    if sampler is None:
        return 0.0
    try:
        return float(sampler(point.x, point.y))
    except TypeError:
        return float(sampler((point.x, point.y)))


def _ground_array(sampler, xy: np.ndarray) -> np.ndarray:
    """Sample the DTM at many XY positions."""
    if sampler is None:
        return np.zeros(len(xy))
    try:
        return np.asarray(sampler(xy), dtype=float).reshape(-1)
    except Exception:
        return np.array([float(sampler(float(x), float(y))) for x, y in xy], dtype=float)


def _plausibility(footprint: Polygon, roof_xyz: np.ndarray, ground: float, settings: dict, *, cut_by_boundary: bool) -> dict:
    """Return the configured geometric building plausibility decision."""
    ring = np.asarray(footprint.minimum_rotated_rectangle.exterior.coords[:-1], dtype=float)
    edges = np.linalg.norm(np.roll(ring, -1, axis=0) - ring, axis=1) if len(ring) else np.empty(0)
    width = float(np.min(edges)) if len(edges) else 0.0
    area = float(footprint.area)
    height = float(np.nanpercentile(roof_xyz[:, 2], 95) - ground) if len(roof_xyz) else 0.0
    reasons = []
    if width < float(settings["min_width_m"]):
        reasons.append(f"minimum width {width:.2f} m < {float(settings['min_width_m']):.2f} m")
    if area < float(settings["min_area_m2"]):
        reasons.append(f"area {area:.2f} m2 < {float(settings['min_area_m2']):.2f} m2")
    if height < float(settings["min_height_m"]):
        reasons.append(f"height {height:.2f} m < {float(settings['min_height_m']):.2f} m")
    return {
        "decision": "rejected" if reasons else "accepted",
        "reason": "; ".join(reasons),
        "min_width_m": width,
        "area_m2": area,
        "height_above_dtm_m": height,
        "cut_by_boundary": bool(cut_by_boundary),
    }


def _roof_subset(points: PointSet, planes: list[RoofPlane]) -> tuple[np.ndarray, list[RoofPlane]]:
    classes = np.asarray(points.classes) if points.classes is not None else np.empty(0, dtype=int)
    selected = np.isin(classes, (ROOF_INCLINED, ROOF_FLAT)) if len(classes) == len(points) else np.zeros(len(points), dtype=bool)
    if not selected.any():
        for plane in planes:
            indices = np.asarray(plane.inliers, dtype=int)
            selected[indices[(indices >= 0) & (indices < len(points))]] = True
    mapping = np.full(len(points), -1, dtype=int)
    mapping[selected] = np.arange(int(selected.sum()))
    copied = []
    for plane in planes:
        inliers = np.asarray(plane.inliers, dtype=int)
        inliers = inliers[(inliers >= 0) & (inliers < len(mapping))]
        copied.append(RoofPlane(np.asarray(plane.normal), float(plane.d), mapping[inliers][mapping[inliers] >= 0], float(plane.rmse), building_id=plane.building_id, outline=plane.outline, slope_deg=plane.slope_deg))
    return np.asarray(points.xyz)[selected], copied


def _heights(partition: Partition, labels: np.ndarray, planes: list[RoofPlane], default: float) -> np.ndarray:
    result = np.full(len(partition.cells), default, dtype=float)
    for index, cell in enumerate(partition.cells):
        label = int(labels[index]) if index < len(labels) else 0
        if 0 < label <= len(planes):
            point = cell.representative_point()
            plane = planes[label - 1]
            result[index] = -float(plane.normal[0] * point.x + plane.normal[1] * point.y + plane.d) / float(plane.normal[2])
    return result


def _roof_extrema(partition: Partition, labels: np.ndarray, planes: list[RoofPlane], footprint: Polygon, default: float) -> tuple[float, float]:
    """Return eave on the outer boundary and ridge over all labelled roof cells."""
    eave_values: list[float] = []
    ridge_values: list[float] = []
    # Facade-derived footprints are normally a few centimetres outside the roof
    # support polygons.  Evaluate every nearby labelled-plane candidate at each
    # outer vertex so a higher annex cannot hide the main roof's eave.
    for coordinate in np.asarray(footprint.exterior.coords[:-1], dtype=float):
        point = Point(coordinate)
        for plane in planes:
            if plane.outline is not None and not plane.outline.is_empty and plane.outline.distance(point) <= 0.25:
                eave_values.append(-float(coordinate @ np.asarray(plane.normal[:2], dtype=float) + plane.d) / float(plane.normal[2]))
    for index, cell in enumerate(partition.cells):
        label = int(labels[index]) if index < len(labels) else 0
        boundary = cell.boundary.intersection(footprint.exterior)
        if not (0 < label <= len(planes)):
            if not boundary.is_empty:
                eave_values.append(default)
            ridge_values.append(default)
            continue
        plane = planes[label - 1]
        coordinates = np.asarray(cell.exterior.coords[:-1], dtype=float)
        if not len(coordinates):
            ridge_values.append(default)
            continue
        heights = -(coordinates @ np.asarray(plane.normal[:2], dtype=float) + float(plane.d)) / float(plane.normal[2])
        ridge_values.extend(np.asarray(heights, dtype=float))
        for line in getattr(boundary, "geoms", [boundary]):
            if not hasattr(line, "coords"):
                continue
            outer_coordinates = np.asarray(line.coords, dtype=float)
            outer_heights = -(outer_coordinates @ np.asarray(plane.normal[:2], dtype=float) + float(plane.d)) / float(plane.normal[2])
            eave_values.extend(np.asarray(outer_heights, dtype=float))
    eave = float(min(eave_values)) if eave_values else default
    ridge = float(max(ridge_values)) if ridge_values else default
    return eave, ridge


def _partition_options(settings: dict) -> dict:
    """Roof generalisation options read from ``buildings.roof`` with defaults."""
    roof = settings.get("roof", {})
    return {
        "tolerance": float(roof.get("inlier_tolerance_m", 0.08)),
        "simplify_m": float(roof.get("region_simplify_m", 0.35)),
        "min_cell_area": float(roof.get("min_region_area_m2", 1.0)),
        "outline_simplify_m": float(roof.get("outline_simplify_m", 0.3)),
        "step_simplify_m": float(roof.get("step_simplify_m", 0.12)),
        "min_feature_area_m2": float(roof.get("min_feature_area_m2", 6.0)),
        "min_step_m": float(roof.get("min_step_m", 0.35)),
    }


def _lod2_candidate(
    points: PointSet,
    footprint: Polygon,
    planes: list[RoofPlane],
    settings: dict,
    *,
    ground: float,
    started: float,
    raster_resolution: float | None = None,
    low_xyz: np.ndarray | None = None,
    check_xyz: np.ndarray | None = None,
) -> tuple[Partition, object, list[RoofPlane], np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    """Build and validate one roof candidate without choosing a fallback."""
    roof_xyz, label_planes = _roof_subset(points, planes)
    if raster_resolution is not None and len(roof_xyz):
        # 2.5D path: every DSM cell is labelled with its best-fitting plane, so
        # cell heights always come from planes and never from footprint corners.
        partition, cell_labels = partition_from_raster(footprint, roof_xyz, label_planes, resolution=raster_resolution, low_xyz=low_xyz, **_partition_options(settings))
        footprint = partition.footprint or footprint
        labels = LabelResult(cell_labels, 0.0, 0, False, [], 0.0)
    else:
        partition = partition_footprint(footprint, plane_lines(planes, footprint))
        remaining = max(0.0, float(settings["max_seconds_per_building"]) - (time.perf_counter() - started))
        labels = label_cells(
            partition,
            label_planes,
            roof_xyz,
            buffer=float(settings["label_pruning_buffer_m"]),
            complexity_factor=float(settings["complexity_factor"]),
            time_limit=remaining,
        )
    try:
        vertices, faces, roles = extrude_lod22(partition, labels.labels, label_planes, footprint, ground)
    except ValueError:
        if raster_resolution is None or not len(roof_xyz):
            raise
        # An unclippable cell ring: retry the 2.5D partition without ridge snapping.
        partition, cell_labels = partition_from_raster(footprint, roof_xyz, label_planes, resolution=raster_resolution, low_xyz=low_xyz, ridge_snap_m=0.0, **_partition_options(settings))
        labels = LabelResult(cell_labels, 0.0, 0, False, [], 0.0)
        footprint = partition.footprint or footprint
        vertices, faces, roles = extrude_lod22(partition, labels.labels, label_planes, footprint, ground)
    on_raster = raster_resolution is not None and len(roof_xyz) > 0
    exclude = partition.feature_geometry.buffer(0.3) if on_raster and getattr(partition, "feature_geometry", None) is not None else None
    report = validate_model(vertices, faces, check_xyz if on_raster and check_xyz is not None and len(check_xyz) else roof_xyz, footprint, vertical=on_raster, exclude=exclude)
    return partition, labels, label_planes, roof_xyz, vertices, faces, roles, report


def _robust_candidate(points, footprint, planes, settings, **kwargs):
    """Build a roof candidate; retry with coarser step tolerances when it is not watertight."""
    result = _lod2_candidate(points, footprint, planes, settings, **kwargs)
    roof = settings.get("roof", {})
    for extra in (0.08, 0.2):
        if result[-1]["status"] == "ok" or kwargs.get("raster_resolution") is None:
            break
        coarser = dict(settings)
        coarser["roof"] = {**roof, "step_simplify_m": float(roof.get("step_simplify_m", 0.12)) + extra}
        result = _lod2_candidate(points, footprint, planes, coarser, **kwargs)
    return result


def reconstruct_building(
    instance_points: PointSet,
    building_id,
    *,
    dtm_sampler,
    config=None,
    dsm_roof_points: np.ndarray | None = None,
    dsm_cell_area_m2: float = 0.01,
    cut_by_boundary: bool = False,
) -> BuildingModel:
    """Reconstruct one building, returning a failed model instead of raising on bad data."""
    timings: dict[str, float] = {}
    started = time.perf_counter()
    try:
        cfg = config or load_drone_config()
        settings = cfg["buildings"]
        xyz = np.asarray(instance_points.xyz, dtype=float)
        if len(xyz) < 12:
            return _failed(building_id, "not enough points", timings)
        stage = time.perf_counter()
        facades = detect_facade_planes(instance_points, distance=float(settings["ransac_distance_m"]), min_points=max(20, min(100, len(xyz) // 8)), max_planes=16)
        footprint, _source = footprint_from_facades(facades, xyz[:, :2])
        timings["footprint"] = time.perf_counter() - stage
        if footprint.is_empty or footprint.area <= 0.1:
            return _failed(building_id, "no usable footprint", timings)
        raster_xyz = np.asarray(dsm_roof_points, dtype=float) if dsm_roof_points is not None else np.empty((0, 3), dtype=float)
        raster_xyz = raster_xyz[np.isfinite(raster_xyz).all(axis=1)]
        all_xyz = raster_xyz
        inner = footprint.buffer(-0.3)
        if inner.is_empty:
            inner = footprint.buffer(-0.05)
        if len(raster_xyz):
            raster_xyz = raster_xyz[np.fromiter((inner.covers(Point(row[0], row[1])) for row in raster_xyz), dtype=bool, count=len(raster_xyz))]
        roof_cfg = settings.get("roof", {"min_plane_area_m2": 4.0, "inlier_tolerance_m": 0.08})
        check_xyz = all_xyz  # unclipped DSM: validates candidates that move the outline
        low_xyz = np.empty((0, 3), dtype=float)
        if len(raster_xyz):
            above = raster_xyz[:, 2] - _ground_array(dtm_sampler, raster_xyz[:, :2])
            low = above < float(roof_cfg.get("min_roof_height_m", 1.5))
            low_xyz, raster_xyz = raster_xyz[low], raster_xyz[~low]
            check_xyz = all_xyz[_ground_array(dtm_sampler, all_xyz[:, :2]) + float(roof_cfg.get("min_roof_height_m", 1.5)) <= all_xyz[:, 2]]
        roof_source = instance_points
        if len(raster_xyz):
            roof_source = PointSet(raster_xyz, np.full(len(raster_xyz), ROOF_FLAT, dtype=np.uint16))
        ground = _ground_value(dtm_sampler, footprint)
        plausibility_cfg = settings.get("plausibility", {"min_width_m": 2.5, "min_area_m2": 10.0, "min_height_m": 2.5})
        plausibility = _plausibility(footprint, np.asarray(roof_source.xyz, dtype=float), ground, plausibility_cfg, cut_by_boundary=cut_by_boundary)
        if plausibility["decision"] == "rejected":
            timings["total"] = time.perf_counter() - started
            return BuildingModel(building_id, "1.2", np.empty((0, 3)), np.empty((0, 3), dtype=int), np.empty(0, dtype=np.uint8), "failed", f"plausibility rejected: {plausibility['reason']}", heights={"timings": timings, "plausibility": plausibility})
        region = settings["region_growing"]
        stage = time.perf_counter()
        planes = detect_raster_roof_planes(
            raster_xyz,
            min_plane_area_m2=float(roof_cfg["min_plane_area_m2"]),
            inlier_tolerance_m=float(roof_cfg["inlier_tolerance_m"]),
            cell_area_m2=float(dsm_cell_area_m2),
            merge_normal_deg=float(roof_cfg.get("merge_normal_deg", 4.0)),
            merge_offset_m=float(roof_cfg.get("merge_offset_m", 0.15)),
        ) if len(raster_xyz) else []
        if not planes:
            roof_source = instance_points
            planes = detect_roof_planes(instance_points, k=int(region["k"]), epsilon=float(region["epsilon_m"]), min_points=min(int(region["min_points"]), max(20, len(xyz) // 8)), normal_angle_deg=float(region["normal_angle_deg"]))
        raster_res = float(np.sqrt(dsm_cell_area_m2)) if (len(raster_xyz) and roof_source is not instance_points) else None
        timings["roof_planes"] = time.perf_counter() - stage
        if not planes:
            return _failed(building_id, "no roof planes", timings)
        minimum_plane = float(plausibility_cfg.get("min_plane_area_m2", 4.0))
        if raster_res is not None:
            largest = max(len(plane.inliers) for plane in planes) * float(dsm_cell_area_m2)
            if largest < minimum_plane:
                plausibility = {**plausibility, "decision": "rejected", "reason": f"largest roof plane {largest:.2f} m2 < {minimum_plane:.2f} m2"}
                timings["total"] = time.perf_counter() - started
                return BuildingModel(building_id, "1.2", np.empty((0, 3)), np.empty((0, 3), dtype=int), np.empty(0, dtype=np.uint8), "failed", f"plausibility rejected: {plausibility['reason']}", heights={"timings": timings, "plausibility": plausibility})
        stage = time.perf_counter()
        baseline = _robust_candidate(roof_source, footprint, planes, settings, ground=ground, started=started, raster_resolution=raster_res, low_xyz=low_xyz, check_xyz=check_xyz)
        timings["labeling"] = time.perf_counter() - stage
        reg_settings = regularize_settings(settings.get("regularize"))
        regular_footprint, footprint_report = regularize_footprint(footprint, reg_settings)
        on_raster = raster_res is not None
        # DSM planes are already exact: ridges are intersections of fitted planes and
        # therefore horizontal.  Gradient snapping would only shift them off the DSM, so on
        # the raster path only the footprint is regularised.
        regular_planes = planes if on_raster else regularize_roof_planes(planes, regular_footprint, xyz=xyz, settings=reg_settings)
        stage = time.perf_counter()
        regular = _robust_candidate(roof_source, regular_footprint, regular_planes, settings, ground=ground, started=started, raster_resolution=raster_res, low_xyz=low_xyz, check_xyz=check_xyz)
        timings["regularization"] = time.perf_counter() - stage
        baseline_report, regular_report = baseline[-1], regular[-1]
        use_regular = regular_report["status"] == "ok"
        regular_reason = str(footprint_report["reason"])
        baseline_rmse, regular_rmse = baseline_report["rmse"], regular_report["rmse"]
        if use_regular and baseline_report["status"] == "ok" and baseline_rmse is not None and regular_rmse is not None:
            if regular_rmse > baseline_rmse + reg_settings["max_rmse_increase_m"]:
                use_regular = False
                regular_reason = "roof RMSE guard retained original model"
        elif not use_regular:
            regular_reason = f"regularized model invalid: {regular_report['reason']}"
        selected = regular if use_regular else baseline
        partition, labels, label_planes, roof_xyz, vertices, faces, roles, report = selected
        regularization_report = {
            "footprint_vertices_before": footprint_report["vertices_before"],
            "footprint_vertices_after": footprint_report["vertices_after"] if use_regular else footprint_report["vertices_before"],
            "footprint_iou": footprint_report["iou"],
            "roof_rmse_before": baseline_rmse,
            "roof_rmse_after": regular_rmse,
            "regularized": use_regular,
            "reason": regular_reason if not use_regular else "regularized",
        }
        logger.info(
            "building %s regularization: footprint vertices %s/%s, IoU %.3f, roof RMSE %s/%s, regularized=%s (%s)",
            building_id,
            regularization_report["footprint_vertices_before"],
            regularization_report["footprint_vertices_after"],
            float(regularization_report["footprint_iou"]),
            baseline_rmse,
            regular_rmse,
            use_regular,
            regularization_report["reason"],
        )
        selected_footprint = (regular_footprint if use_regular else footprint)
        selected_footprint = partition.footprint or selected_footprint
        roof_height = _heights(partition, labels.labels, label_planes, float(np.percentile(roof_xyz[:, 2], 50)))
        eave, ridge = _roof_extrema(partition, labels.labels, label_planes, selected_footprint, float(np.percentile(roof_xyz[:, 2], 50)))
        height_info = {
            "ground": ground,
            "eave": eave,
            "ridge": ridge,
            "timings": timings,
            "regularization": regularization_report,
            "plausibility": plausibility,
            "roof_planes": [
                {"support": int(len(plane.inliers)), "normal": np.asarray(plane.normal, dtype=float).round(6).tolist(), "rmse": float(plane.rmse)}
                for index, plane in enumerate(label_planes, start=1) if index in set(int(item) for item in np.asarray(labels.labels, dtype=int))
            ],
            "rmse_lod2": report.get("rmse"),
            "rmse_all": report.get("rmse_all", report.get("rmse")),
            "superstructures": {
                "count": len(getattr(partition, "superstructures", [])),
                "area_m2": float(sum(item["area_m2"] for item in getattr(partition, "superstructures", []))),
                "items": list(getattr(partition, "superstructures", [])),
            },
            "partition_cells": len(partition.cells),
            "labels": np.asarray(labels.labels, dtype=int).tolist(),
            "roof_source": "dsm" if len(raster_xyz) else "point_cloud",
        }
        if not labels.timed_out and report["status"] == "ok":
            height_info["timings"]["total"] = time.perf_counter() - started
            return BuildingModel(building_id, "2.2", vertices, faces, roles, "ok", "", report["rmse"], report["coverage"], height_info)
        if labels.timed_out:
            lod2_reason = "labeling timed out"
        else:
            lod2_reason = report["reason"]
        vertices, faces, roles = extrude_lod13(partition, roof_height, ground)
        report = validate_model(vertices, faces, roof_xyz, selected_footprint)
        if report["status"] == "ok":
            height_info["timings"]["total"] = time.perf_counter() - started
            return BuildingModel(building_id, "1.3", vertices, faces, roles, "lod1_fallback", lod2_reason, report["rmse"], report["coverage"], height_info)
        top = float(np.percentile(roof_xyz[:, 2], 50)) if len(roof_xyz) else ground + 3.0
        vertices, faces, roles = extrude_lod1(selected_footprint, top, ground)
        report = validate_model(vertices, faces, roof_xyz, selected_footprint)
        if report["status"] == "ok":
            height_info["timings"]["total"] = time.perf_counter() - started
            return BuildingModel(building_id, "1.2", vertices, faces, roles, "lod1_fallback", f"LoD2 invalid: {lod2_reason}", report["rmse"], report["coverage"], height_info)
        return _failed(building_id, f"LoD fallbacks invalid: {report['reason']}", timings)
    except Exception as exc:
        return _failed(building_id, f"reconstruction failed: {exc}", timings)
