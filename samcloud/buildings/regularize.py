"""Conservative LoD2 footprint and roof-plane regularisation."""

from __future__ import annotations

import math
from collections.abc import Mapping

import numpy as np
from shapely.geometry import Polygon

from samcloud.drone.contracts import RoofPlane
from samcloud.ortho.vector.regularize import dominant_direction, regularize_polygon


DEFAULT_REGULARIZE: dict[str, float] = {
    "angle_tol_deg": 12.0,
    "merge_offset_m": 0.3,
    "min_edge_m": 0.5,
    "ridge_tol_deg": 5.0,
    "flat_slope_deg": 5.0,
    "eave_tol_m": 0.15,
    "min_footprint_iou": 0.92,
    "max_rmse_increase_m": 0.03,
}


def regularize_settings(value: Mapping[str, object] | None) -> dict[str, float]:
    """Return complete, numeric building-regularisation settings."""
    result = DEFAULT_REGULARIZE.copy()
    if value:
        for key in result:
            if key in value:
                result[key] = float(value[key])
    return result


def footprint_iou(first: Polygon, second: Polygon) -> float:
    """Return the footprint intersection-over-union, including empty guards."""
    if first.is_empty or second.is_empty:
        return 0.0
    union = first.union(second).area
    return float(first.intersection(second).area / union) if union > 1e-10 else 0.0


def regularize_footprint(footprint: Polygon, settings: Mapping[str, object] | None = None) -> tuple[Polygon, dict[str, object]]:
    """Snap a footprint conservatively and retain it only when its IoU is safe."""
    cfg = regularize_settings(settings)
    before = max(0, len(footprint.exterior.coords) - 1) if not footprint.is_empty else 0
    if footprint.is_empty or before < 3:
        return footprint, {"vertices_before": before, "vertices_after": before, "iou": 1.0, "reason": "empty footprint"}
    direction = dominant_direction([footprint])
    candidate = regularize_polygon(
        footprint,
        direction=direction,
        angle_tolerance_deg=cfg["angle_tol_deg"],
        min_edge_length=cfg["min_edge_m"],
    )
    iou = footprint_iou(footprint, candidate)
    if candidate.is_empty or not candidate.is_valid or iou < cfg["min_footprint_iou"]:
        return footprint, {
            "vertices_before": before,
            "vertices_after": before,
            "iou": iou,
            "reason": "IoU guard retained original footprint",
        }
    return candidate, {
        "vertices_before": before,
        "vertices_after": len(candidate.exterior.coords) - 1,
        "iou": iou,
        "reason": "regularized",
    }


def _normal_from_gradient(gradient: np.ndarray, intercept: float) -> tuple[np.ndarray, float]:
    vector = np.array((-gradient[0], -gradient[1], 1.0), dtype=float)
    scale = float(np.linalg.norm(vector))
    return vector / scale, -float(intercept) / scale


def _plane_points(plane: RoofPlane, xyz: np.ndarray | None) -> np.ndarray:
    if xyz is None or len(xyz) == 0:
        return np.empty((0, 3), dtype=float)
    indices = np.asarray(plane.inliers, dtype=int)
    indices = indices[(indices >= 0) & (indices < len(xyz))]
    return xyz[indices]


def _intercept(points: np.ndarray, gradient: np.ndarray, fallback: float) -> float:
    if not len(points):
        return fallback
    return float(np.median(points[:, 2] - points[:, :2] @ gradient))


def _gradient(plane: RoofPlane) -> np.ndarray:
    normal = np.asarray(plane.normal, dtype=float)
    return -normal[:2] / max(float(normal[2]), 1e-12)


def regularize_roof_planes(
    planes: list[RoofPlane],
    footprint: Polygon,
    *,
    xyz: np.ndarray | None = None,
    settings: Mapping[str, object] | None = None,
    collapse_dominant: bool = True,
) -> list[RoofPlane]:
    """Align roof gradients to footprint axes while preserving fitted elevations.

    A plane is only snapped when its gradient already lies within the configured
    angular tolerance of an axis.  Opposing, similarly sloped planes share a
    gradient magnitude, making their ridge horizontal after refitting.
    """
    if not planes or footprint.is_empty:
        return list(planes)
    cfg = regularize_settings(settings)
    axis_angle = dominant_direction([footprint])
    axes = [np.array((math.cos(axis_angle), math.sin(axis_angle))), np.array((-math.sin(axis_angle), math.cos(axis_angle)))]
    tolerance = math.radians(cfg["angle_tol_deg"])
    support = np.asarray([len(plane.inliers) for plane in planes], dtype=float)
    # Drone classifications can leave small facade fragments and ridge outliers
    # in the roof set.  When one fitted plane clearly explains the roof support,
    # retaining every fragment produces an artificial apex fan.  Keep that
    # dominant plane as the conservative, single-plane roof hypothesis.
    if collapse_dominant and support.sum() > 0 and support.max() / support.sum() >= 0.8:
        planes = [planes[int(np.argmax(support))]]
    records: list[dict[str, object]] = []
    for plane in planes:
        gradient = _gradient(plane)
        magnitude = float(np.linalg.norm(gradient))
        points = _plane_points(plane, xyz)
        fallback = -float(plane.d) / max(float(plane.normal[2]), 1e-12)
        if math.degrees(math.atan(magnitude)) < cfg["flat_slope_deg"]:
            snapped = np.zeros(2, dtype=float)
        elif magnitude <= 1e-12:
            snapped = gradient
        else:
            unit = gradient / magnitude
            choices = [axis for axis in axes] + [-axis for axis in axes]
            target = max(choices, key=lambda axis: float(unit @ axis))
            error = math.acos(float(np.clip(unit @ target, -1.0, 1.0)))
            snapped = target * magnitude if error <= tolerance else gradient
        records.append({"plane": plane, "gradient": snapped, "points": points, "fallback": fallback})
    ridge_tolerance = math.radians(cfg["ridge_tol_deg"])
    for left, first in enumerate(records):
        gradient_a = np.asarray(first["gradient"], dtype=float)
        length_a = float(np.linalg.norm(gradient_a))
        if length_a <= 1e-10:
            continue
        for second in records[left + 1:]:
            gradient_b = np.asarray(second["gradient"], dtype=float)
            length_b = float(np.linalg.norm(gradient_b))
            if length_b <= 1e-10:
                continue
            opposing = float(gradient_a @ gradient_b) / (length_a * length_b) <= -math.cos(ridge_tolerance)
            slope_delta = abs(math.atan(length_a) - math.atan(length_b))
            if not opposing or slope_delta > ridge_tolerance:
                continue
            average = (length_a + length_b) * 0.5
            first["gradient"] = gradient_a / length_a * average
            second["gradient"] = gradient_b / length_b * average
            break
    result: list[RoofPlane] = []
    for record in records:
        original = record["plane"]
        gradient = np.asarray(record["gradient"], dtype=float)
        intercept = _intercept(np.asarray(record["points"], dtype=float), gradient, float(record["fallback"]))
        normal, d = _normal_from_gradient(gradient, intercept)
        result.append(RoofPlane(normal, d, np.asarray(original.inliers, dtype=int), float(original.rmse), building_id=original.building_id, outline=original.outline, slope_deg=math.degrees(math.atan(float(np.linalg.norm(gradient))))))
    return result
