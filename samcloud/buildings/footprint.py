"""Facade-plane extraction and footprint reconstruction."""

from __future__ import annotations

import math

import numpy as np
from scipy.spatial import cKDTree
from shapely import concave_hull, make_valid
from shapely.geometry import LineString, MultiPoint, Polygon

from samcloud.drone.contracts import FacadePlane, PointSet
from samcloud.ortho.vector.regularize import dominant_direction, regularize_polygon


def _normals(xyz: np.ndarray, supplied: np.ndarray | None, neighbours: int = 16) -> np.ndarray:
    """Return unit local-PCA normals, preserving usable supplied normals."""
    if supplied is not None and len(supplied) == len(xyz):
        result = np.asarray(supplied, dtype=float).copy()
        length = np.linalg.norm(result, axis=1)
        valid = length > 1e-9
        result[valid] /= length[valid, None]
        if valid.all():
            return result
    else:
        result = np.zeros_like(xyz, dtype=float)
        valid = np.zeros(len(xyz), dtype=bool)
    if len(xyz) < 3:
        return result
    count = min(neighbours, len(xyz))
    indices = cKDTree(xyz).query(xyz, k=count, workers=-1)[1]
    local = xyz[indices]
    local -= local.mean(axis=1, keepdims=True)
    covariance = np.einsum("nki,nkj->nij", local, local) / max(count - 1, 1)
    values, vectors = np.linalg.eigh(covariance)
    result[~valid] = vectors[~valid, :, 0]
    length = np.linalg.norm(result, axis=1)
    return result / np.maximum(length[:, None], 1e-12)


def _line_fit(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fit a 2-D total-least-squares line as origin and unit normal."""
    origin = points.mean(axis=0)
    _values, vectors = np.linalg.eigh(np.cov((points - origin).T, bias=True))
    normal = vectors[:, 0]
    normal /= np.linalg.norm(normal)
    if normal[0] < -1e-12 or (abs(normal[0]) <= 1e-12 and normal[1] < 0):
        normal = -normal
    return origin, normal


def _segment(origin: np.ndarray, normal: np.ndarray, xy: np.ndarray) -> LineString:
    direction = np.asarray((-normal[1], normal[0]))
    values = (xy - origin) @ direction
    return LineString([tuple(origin + values.min() * direction), tuple(origin + values.max() * direction)])


def _merge_facades(facades: list[FacadePlane], xyz: np.ndarray) -> list[FacadePlane]:
    merged: list[FacadePlane] = []
    for facade in facades:
        normal = facade.normal[:2]
        for index, existing in enumerate(merged):
            other = existing.normal[:2]
            aligned = abs(float(np.dot(normal, other)))
            if aligned < math.cos(math.radians(3.0)):
                continue
            sign = 1.0 if float(np.dot(normal, other)) >= 0.0 else -1.0
            if abs(facade.d * sign - existing.d) >= 0.1:
                continue
            inliers = np.unique(np.concatenate((existing.inliers, facade.inliers)))
            origin, fitted_normal = _line_fit(xyz[inliers, :2])
            d = -float(np.dot(fitted_normal, origin))
            segment = _segment(origin, fitted_normal, xyz[inliers, :2])
            merged[index] = FacadePlane(
                normal=np.array((fitted_normal[0], fitted_normal[1], 0.0)), d=d,
                inliers=inliers, rmse=float(np.mean(np.abs(xyz[inliers, :2] @ fitted_normal + d))),
                segment=segment, z_min=float(xyz[inliers, 2].min()), z_max=float(xyz[inliers, 2].max()),
            )
            break
        else:
            merged.append(facade)
    return merged


def detect_facade_planes(ps: PointSet, *, distance: float, min_points: int, max_planes: int) -> list[FacadePlane]:
    """Extract vertical facade planes by sequential, 2-D RANSAC fitting."""
    xyz = np.asarray(ps.xyz, dtype=float)
    if len(xyz) < min_points:
        return []
    normals = _normals(xyz, ps.normals)
    candidates = np.flatnonzero(np.abs(normals[:, 2]) < 0.3)
    remaining = candidates.copy()
    rng = np.random.default_rng(73451)
    found: list[FacadePlane] = []
    trials = min(1_000, max(150, len(remaining) * 2))
    while len(remaining) >= min_points and len(found) < max_planes:
        xy = xyz[remaining, :2]
        best: np.ndarray | None = None
        best_count = 0
        for _ in range(trials):
            pair = rng.choice(len(remaining), size=2, replace=False)
            direction = xy[pair[1]] - xy[pair[0]]
            scale = float(np.linalg.norm(direction))
            if scale < 1e-6:
                continue
            normal = np.array((-direction[1], direction[0])) / scale
            residual = np.abs((xy - xy[pair[0]]) @ normal)
            choice = residual <= distance
            count = int(choice.sum())
            if count > best_count:
                best, best_count = choice, count
        if best is None or best_count < min_points:
            break
        provisional = remaining[best]
        origin, normal = _line_fit(xyz[provisional, :2])
        d = -float(normal @ origin)
        selected = remaining[np.abs(xyz[remaining, :2] @ normal + d) <= distance]
        if len(selected) < min_points:
            break
        origin, normal = _line_fit(xyz[selected, :2])
        d = -float(normal @ origin)
        residual = np.abs(xyz[selected, :2] @ normal + d)
        found.append(FacadePlane(
            normal=np.array((normal[0], normal[1], 0.0)), d=d, inliers=selected,
            rmse=float(np.sqrt(np.mean(residual ** 2))), segment=_segment(origin, normal, xyz[selected, :2]),
            z_min=float(xyz[selected, 2].min()), z_max=float(xyz[selected, 2].max()),
        ))
        keep = np.abs(xyz[remaining, :2] @ normal + d) > distance
        remaining = remaining[keep]
    return _merge_facades(found, xyz)


def _as_outline(value: object | None) -> Polygon | None:
    if value is None:
        return None
    if isinstance(value, Polygon):
        return value
    points = np.asarray(value, dtype=float)
    if points.ndim != 2 or points.shape[0] < 3 or points.shape[1] < 2:
        return None
    hull = concave_hull(MultiPoint(points[:, :2]), ratio=0.35, allow_holes=False)
    if hull.is_empty:
        return None
    if not isinstance(hull, Polygon):
        hull = MultiPoint(points[:, :2]).convex_hull
    return hull.buffer(0.02).simplify(0.02, preserve_topology=True)


def _intersection(first: LineString, second: LineString) -> tuple[float, float] | None:
    a, b = np.asarray(first.coords[0]), np.asarray(first.coords[-1])
    c, d = np.asarray(second.coords[0]), np.asarray(second.coords[-1])
    direction_a, direction_b = b - a, d - c
    determinant = float(direction_a[0] * direction_b[1] - direction_a[1] * direction_b[0])
    if abs(determinant) < 1e-8:
        return None
    factor = float(((c[0] - a[0]) * direction_b[1] - (c[1] - a[1]) * direction_b[0]) / determinant)
    point = a + factor * direction_a
    return float(point[0]), float(point[1])


def footprint_from_facades(facades: list[FacadePlane], roof_outline=None) -> tuple[Polygon, str]:
    """Reconstruct a regularised footprint from facade traces, falling back to roof support."""
    outline = _as_outline(roof_outline)
    usable = [item for item in facades if item.segment is not None and item.segment.length > 0.05]
    if len(usable) < 3:
        if outline is None:
            return Polygon(), "roof_outline"
        return regularize_polygon(outline, direction=dominant_direction([outline])), "roof_outline"
    centers = np.asarray([item.segment.interpolate(0.5, normalized=True).coords[0] for item in usable])
    centroid = centers.mean(axis=0)
    usable.sort(key=lambda item: math.atan2(item.segment.centroid.y - centroid[1], item.segment.centroid.x - centroid[0]))
    corners = [_intersection(first.segment, second.segment) for first, second in zip(usable, usable[1:] + usable[:1], strict=False)]
    valid = [corner for corner in corners if corner is not None]
    candidate = Polygon(valid) if len(valid) >= 3 else Polygon()
    mixed = len(valid) != len(usable) or not candidate.is_valid or candidate.area < 0.05
    if candidate.is_empty or not candidate.is_valid or candidate.area < 0.05:
        if outline is None:
            return Polygon(), "roof_outline"
        return regularize_polygon(outline, direction=dominant_direction([outline])), "roof_outline"
    candidate = make_valid(candidate)
    if not isinstance(candidate, Polygon):
        candidate = max((part for part in getattr(candidate, "geoms", []) if isinstance(part, Polygon)), key=lambda p: p.area, default=Polygon())
    if outline is not None and (not outline.buffer(0.5).contains(candidate) or candidate.area > outline.area * 1.3):
        candidate = candidate.intersection(outline.buffer(0.1))
        mixed = True
    if not isinstance(candidate, Polygon) or candidate.is_empty:
        return regularize_polygon(outline, direction=dominant_direction([outline])), "roof_outline" if outline else "roof_outline"
    regular = regularize_polygon(candidate, direction=dominant_direction([item.segment for item in usable]))
    return regular, "mixed" if mixed else "facades"
