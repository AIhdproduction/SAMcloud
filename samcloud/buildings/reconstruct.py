"""End-to-end, failure-tolerant building reconstruction."""

from __future__ import annotations

import time

import numpy as np
from shapely.geometry import MultiPoint, Point, Polygon

from samcloud.buildings.extrude import extrude_lod1, extrude_lod13, extrude_lod22
from samcloud.buildings.footprint import detect_facade_planes, footprint_from_facades
from samcloud.buildings.labeling import label_cells
from samcloud.buildings.partition import Partition, partition_footprint, plane_lines
from samcloud.buildings.roof_planes import detect_roof_planes
from samcloud.buildings.validate import validate_model
from samcloud.drone.contracts import BuildingModel, PointSet, ROOF_FLAT, ROOF_INCLINED, RoofPlane, load_drone_config


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


def reconstruct_building(instance_points: PointSet, building_id, *, dtm_sampler, config=None) -> BuildingModel:
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
        region = settings["region_growing"]
        stage = time.perf_counter()
        planes = detect_roof_planes(instance_points, k=int(region["k"]), epsilon=float(region["epsilon_m"]), min_points=min(int(region["min_points"]), max(20, len(xyz) // 8)), normal_angle_deg=float(region["normal_angle_deg"]))
        timings["roof_planes"] = time.perf_counter() - stage
        if not planes:
            return _failed(building_id, "no roof planes", timings)
        partition = partition_footprint(footprint, plane_lines(planes, footprint))
        roof_xyz, label_planes = _roof_subset(instance_points, planes)
        stage = time.perf_counter()
        remaining = max(0.0, float(settings["max_seconds_per_building"]) - (time.perf_counter() - started))
        labels = label_cells(partition, label_planes, roof_xyz, buffer=float(settings["label_pruning_buffer_m"]), complexity_factor=float(settings["complexity_factor"]), time_limit=remaining)
        timings["labeling"] = time.perf_counter() - stage
        ground = _ground_value(dtm_sampler, footprint)
        roof_height = _heights(partition, labels.labels, label_planes, float(np.percentile(roof_xyz[:, 2], 50)))
        eave, ridge = _roof_extrema(partition, labels.labels, label_planes, footprint, float(np.percentile(roof_xyz[:, 2], 50)))
        height_info = {"ground": ground, "eave": eave, "ridge": ridge, "timings": timings}
        if not labels.timed_out:
            vertices, faces, roles = extrude_lod22(partition, labels.labels, label_planes, footprint, ground)
            report = validate_model(vertices, faces, roof_xyz, footprint)
            if report["status"] == "ok":
                height_info["timings"]["total"] = time.perf_counter() - started
                return BuildingModel(building_id, "2.2", vertices, faces, roles, "ok", "", report["rmse"], report["coverage"], height_info)
            lod2_reason = report["reason"]
        else:
            lod2_reason = "labeling timed out"
        vertices, faces, roles = extrude_lod13(partition, roof_height, ground)
        report = validate_model(vertices, faces, roof_xyz, footprint)
        if report["status"] == "ok":
            height_info["timings"]["total"] = time.perf_counter() - started
            return BuildingModel(building_id, "1.3", vertices, faces, roles, "lod1_fallback", lod2_reason, report["rmse"], report["coverage"], height_info)
        top = float(np.percentile(roof_xyz[:, 2], 50)) if len(roof_xyz) else ground + 3.0
        vertices, faces, roles = extrude_lod1(footprint, top, ground)
        report = validate_model(vertices, faces, roof_xyz, footprint)
        if report["status"] == "ok":
            height_info["timings"]["total"] = time.perf_counter() - started
            return BuildingModel(building_id, "1.2", vertices, faces, roles, "lod1_fallback", f"LoD2 invalid: {lod2_reason}", report["rmse"], report["coverage"], height_info)
        return _failed(building_id, f"LoD fallbacks invalid: {report['reason']}", timings)
    except Exception as exc:
        return _failed(building_id, f"reconstruction failed: {exc}", timings)
