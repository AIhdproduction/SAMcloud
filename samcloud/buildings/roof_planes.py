"""Region-growing roof plane detection."""

from __future__ import annotations

import math

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree
from shapely import concave_hull
from shapely.geometry import MultiPoint, Polygon

from samcloud.drone.contracts import PointSet, ROOF_FLAT, ROOF_INCLINED, RoofPlane
from samcloud.buildings.footprint import _normals


def _fit_plane(points: np.ndarray) -> tuple[np.ndarray, float, np.ndarray]:
    """Fit an upward-facing total-least-squares plane."""
    center = points.mean(axis=0)
    _values, vectors = np.linalg.eigh(np.cov((points - center).T, bias=True))
    normal = vectors[:, 0]
    if normal[2] < 0:
        normal = -normal
    normal /= np.linalg.norm(normal)
    d = -float(normal @ center)
    return normal, d, np.abs(points @ normal + d)


def _outline(points: np.ndarray) -> Polygon:
    geometry = concave_hull(MultiPoint(points[:, :2]), ratio=0.45, allow_holes=False)
    if isinstance(geometry, Polygon):
        return geometry.buffer(0)
    hull = MultiPoint(points[:, :2]).convex_hull
    return hull if isinstance(hull, Polygon) else Polygon()


def _merge_regions(regions: list[np.ndarray], xyz: np.ndarray, epsilon: float, angle: float) -> list[np.ndarray]:
    """Merge nearby coplanar grown regions until stable."""
    changed = True
    cosine = math.cos(math.radians(angle))
    while changed:
        changed = False
        models = [_fit_plane(xyz[item])[:2] for item in regions]
        for first in range(len(regions)):
            for second in range(first + 1, len(regions)):
                normal_a, d_a = models[first]
                normal_b, d_b = models[second]
                if float(normal_a @ normal_b) < cosine:
                    continue
                points_b = xyz[regions[second]]
                points_a = xyz[regions[first]]
                if max(float(np.median(np.abs(points_b @ normal_a + d_a))), float(np.median(np.abs(points_a @ normal_b + d_b)))) > epsilon:
                    continue
                if _outline(points_a).distance(_outline(points_b)) > epsilon * 4.0:
                    continue
                regions[first] = np.unique(np.concatenate((regions[first], regions[second])))
                regions.pop(second)
                changed = True
                break
            if changed:
                break
    return regions


def detect_roof_planes(
    ps: PointSet, *, k: int, epsilon: float, min_points: int, normal_angle_deg: float
) -> list[RoofPlane]:
    """Detect roof planes with normal-aware KNN region growing."""
    xyz_all = np.asarray(ps.xyz, dtype=float)
    if len(xyz_all) < min_points:
        return []
    classes = np.asarray(ps.classes) if ps.classes is not None else np.empty(0, dtype=int)
    normals_all = _normals(xyz_all, ps.normals, neighbours=max(8, k))
    roof_mask = np.isin(classes, (ROOF_INCLINED, ROOF_FLAT)) if len(classes) == len(xyz_all) else np.zeros(len(xyz_all), dtype=bool)
    # Building-class clouds commonly retain walls and roofs under the same LAS
    # code.  Without roof-specific classes, admitting vertical facade points here
    # creates spurious roof planes and unstable footprint partitions.
    selected = np.flatnonzero(roof_mask) if roof_mask.any() else np.flatnonzero(np.abs(normals_all[:, 2]) >= 0.3)
    xyz = xyz_all[selected]
    if len(xyz) < min_points:
        return []
    normals = normals_all[selected]
    if len(normals):
        normals[normals[:, 2] < 0] *= -1.0
    neighbours = cKDTree(xyz).query(xyz, k=min(max(2, k + 1), len(xyz)), workers=-1)[1]
    if neighbours.ndim == 1:
        neighbours = neighbours[:, None]
    local = xyz[neighbours]
    local -= local.mean(axis=1, keepdims=True)
    covariance = np.einsum("nki,nkj->nij", local, local) / max(local.shape[1] - 1, 1)
    values = np.linalg.eigvalsh(covariance)
    curvature = values[:, 0] / np.maximum(values.sum(axis=1), 1e-12)
    cosine = math.cos(math.radians(normal_angle_deg))
    # This is the region-growing frontier in sparse form.  Computing all KNN
    # compatibility tests at once avoids a Python loop over millions of edges.
    neighbour_ids = neighbours[:, 1:]
    del local, covariance, values
    compatible_normals = np.einsum("nki,ni->nk", normals[neighbour_ids], normals) >= cosine
    offsets = xyz[neighbour_ids] - xyz[:, None, :]
    compatible_distance = np.abs(np.einsum("nki,ni->nk", offsets, normals)) <= epsilon
    compatible = compatible_normals & compatible_distance
    rows = np.repeat(np.arange(len(xyz)), neighbour_ids.shape[1])[compatible.ravel()]
    columns = neighbour_ids.ravel()[compatible.ravel()]
    graph = csr_matrix((np.ones(len(rows), dtype=np.uint8), (rows, columns)), shape=(len(xyz), len(xyz)))
    _count, labels = connected_components(graph, directed=False, return_labels=True)
    regions = [np.flatnonzero(labels == label) for label in np.unique(labels) if np.count_nonzero(labels == label) >= min_points]
    regions = _merge_regions(regions, xyz, epsilon, normal_angle_deg)
    result: list[RoofPlane] = []
    for region in regions:
        if len(region) < min_points:
            continue
        normal, d, residual = _fit_plane(xyz[region])
        slope = math.degrees(math.acos(float(np.clip(normal[2], -1.0, 1.0))))
        result.append(RoofPlane(
            normal=normal, d=d, inliers=selected[region], rmse=float(np.sqrt(np.mean(residual ** 2))),
            outline=_outline(xyz[region]), slope_deg=slope,
        ))
    return result


def merge_coplanar_planes(planes: list[RoofPlane], samples: np.ndarray, *, normal_deg: float = 4.0, offset_m: float = 0.15) -> list[RoofPlane]:
    """Merge planes whose normals differ < ``normal_deg`` and whose heights differ < ``offset_m``.

    One roof face split by noise or by an occluding feature becomes one plane again,
    refitted on the union of both supports.
    """
    planes = list(planes)
    cosine = math.cos(math.radians(normal_deg))
    changed = True
    while changed and len(planes) > 1:
        changed = False
        for first in range(len(planes)):
            for second in range(first + 1, len(planes)):
                a, b = planes[first], planes[second]
                if float(np.asarray(a.normal) @ np.asarray(b.normal)) < cosine:
                    continue
                union = np.unique(np.concatenate((np.asarray(a.inliers, dtype=int), np.asarray(b.inliers, dtype=int))))
                points = samples[union]
                height_a = -(points[:, :2] @ np.asarray(a.normal[:2]) + a.d) / a.normal[2]
                height_b = -(points[:, :2] @ np.asarray(b.normal[:2]) + b.d) / b.normal[2]
                if float(np.mean(np.abs(height_a - height_b))) >= offset_m:
                    continue
                normal, d, residual = _fit_plane(points)
                outline = _outline(points)
                planes[first] = RoofPlane(
                    normal=normal, d=d, inliers=union, rmse=float(np.sqrt(np.mean(residual ** 2))), outline=outline,
                    slope_deg=math.degrees(math.acos(float(np.clip(normal[2], -1.0, 1.0)))),
                )
                planes.pop(second)
                changed = True
                break
            if changed:
                break
    return planes


def detect_raster_roof_planes(
    xyz: np.ndarray,
    *,
    min_plane_area_m2: float = 4.0,
    inlier_tolerance_m: float = 0.08,
    cell_area_m2: float = 0.01,
    max_planes: int = 12,
    merge_normal_deg: float = 4.0,
    merge_offset_m: float = 0.15,
) -> list[RoofPlane]:
    """Fit the dominant roof faces from measured DSM cells.

    The DSM is a regular 2.5D observation, so sequential RANSAC is both more
    stable than point-cloud normal growing and naturally rejects sparse facade
    returns.  ``inliers`` deliberately index ``xyz``; callers can pass the
    same raster support directly to partition labelling and validation.
    """
    samples = np.asarray(xyz, dtype=float)
    samples = samples[np.isfinite(samples).all(axis=1)]
    if len(samples) < 12 or cell_area_m2 <= 0:
        return []
    minimum = max(12, int(math.ceil(float(min_plane_area_m2) / float(cell_area_m2))))
    if len(samples) < minimum:
        return []
    rng = np.random.default_rng(20261121)
    remaining = np.arange(len(samples), dtype=int)
    result: list[RoofPlane] = []
    for _plane_index in range(max_planes):
        if len(remaining) < minimum:
            break
        active = samples[remaining]
        best: np.ndarray | None = None
        trials = min(640, max(128, len(active) // 8))
        for _trial in range(trials):
            choice = rng.choice(len(active), size=3, replace=False)
            seed = active[choice]
            normal = np.cross(seed[1] - seed[0], seed[2] - seed[0])
            norm = float(np.linalg.norm(normal))
            if norm <= 1e-10:
                continue
            normal /= norm
            if abs(float(normal[2])) < 0.25:
                continue
            d = -float(normal @ seed[0])
            inliers = np.flatnonzero(np.abs(active @ normal + d) <= inlier_tolerance_m)
            if best is None or len(inliers) > len(best):
                best = inliers
        if best is None or len(best) < minimum:
            break
        normal, d, _residual = _fit_plane(active[best])
        refined = np.flatnonzero(np.abs(active @ normal + d) <= inlier_tolerance_m)
        if len(refined) < minimum:
            break
        normal, d, residual = _fit_plane(active[refined])
        refined = np.flatnonzero(np.abs(active @ normal + d) <= inlier_tolerance_m)
        if len(refined) < minimum:
            break
        indices = remaining[refined]
        support = samples[indices]
        outline = _outline(support)
        if outline.is_empty or outline.area < min_plane_area_m2 * 0.5:
            remaining = np.delete(remaining, refined)
            continue
        slope = math.degrees(math.acos(float(np.clip(normal[2], -1.0, 1.0))))
        result.append(RoofPlane(
            normal=normal,
            d=d,
            inliers=indices,
            rmse=float(np.sqrt(np.mean(residual ** 2))),
            outline=outline,
            slope_deg=slope,
        ))
        remaining = np.delete(remaining, refined)
    return merge_coplanar_planes(result, samples, normal_deg=merge_normal_deg, offset_m=merge_offset_m)


def plane_ids(ps: PointSet, roof_planes: list[RoofPlane], *, distance: float = 0.08) -> np.ndarray:
    """Assign each point the nearest accepted roof-plane index, or -1."""
    xyz = np.asarray(ps.xyz, dtype=float)
    result = np.full(len(xyz), -1, dtype=int)
    if not roof_planes:
        return result
    distances = np.column_stack([np.abs(plane.distance(xyz)) for plane in roof_planes])
    best = np.argmin(distances, axis=1)
    accepted = distances[np.arange(len(xyz)), best] <= distance
    result[accepted] = best[accepted]
    return result
