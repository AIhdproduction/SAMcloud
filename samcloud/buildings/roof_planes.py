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
