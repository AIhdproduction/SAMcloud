"""Lightweight geometric checks for reconstructed building meshes."""

from __future__ import annotations

import numpy as np
from shapely.geometry import Polygon
from shapely.ops import unary_union


def _fallback_distances(points: np.ndarray, vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Approximate nearest roof-face distances when trimesh acceleration is unavailable."""
    triangles = vertices[faces]
    normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    lengths = np.linalg.norm(normals, axis=1)
    normals = normals / np.maximum(lengths[:, None], 1e-12)
    values = np.abs(np.einsum("nti,ti->nt", points[:, None, :] - triangles[None, :, 0, :], normals))
    return values.min(axis=1)


def roof_height_rmse(vertices: np.ndarray, roof_faces: np.ndarray, points: np.ndarray, footprint: Polygon, *, edge_margin: float = 0.3, exclude=None) -> float | None:
    """Vertical RMSE between DSM cells and the roof surface, ignoring the outline margin."""
    points = np.asarray(points, dtype=float)
    if len(points) == 0 or len(roof_faces) == 0:
        return None
    inner = footprint.buffer(-edge_margin)
    if not inner.is_empty:
        from shapely import points as shapely_points

        points = points[np.asarray(inner.covers(shapely_points(points[:, :2])), dtype=bool)]
    if len(points) == 0:
        return None
    if exclude is not None and not exclude.is_empty:
        from shapely import points as shapely_points_

        points = points[~np.asarray(exclude.covers(shapely_points_(points[:, :2])), dtype=bool)]
        if len(points) == 0:
            return None
    heights = np.full(len(points), np.nan)
    for face in roof_faces:
        a, b, c = vertices[face]
        den = (b[1] - c[1]) * (a[0] - c[0]) + (c[0] - b[0]) * (a[1] - c[1])
        if abs(den) < 1e-12:
            continue
        index = np.flatnonzero(
            (points[:, 0] >= min(a[0], b[0], c[0]) - 1e-9) & (points[:, 0] <= max(a[0], b[0], c[0]) + 1e-9)
            & (points[:, 1] >= min(a[1], b[1], c[1]) - 1e-9) & (points[:, 1] <= max(a[1], b[1], c[1]) + 1e-9)
        )
        if not len(index):
            continue
        x, y = points[index, 0], points[index, 1]
        l1 = ((b[1] - c[1]) * (x - c[0]) + (c[0] - b[0]) * (y - c[1])) / den
        l2 = ((c[1] - a[1]) * (x - c[0]) + (a[0] - c[0]) * (y - c[1])) / den
        inside = (l1 >= -1e-9) & (l2 >= -1e-9) & (1.0 - l1 - l2 >= -1e-9)
        heights[index[inside]] = (l1 * a[2] + l2 * b[2] + (1.0 - l1 - l2) * c[2])[inside]
    valid = np.isfinite(heights)
    if not valid.any():
        return None
    return float(np.sqrt(np.mean((points[valid, 2] - heights[valid]) ** 2)))


def validate_model(V: np.ndarray, F: np.ndarray, roof_points_xyz: np.ndarray, footprint: Polygon, *, vertical: bool = False, exclude=None) -> dict:
    """Check a mesh and report topology, roof-fit RMSE, and footprint coverage."""
    vertices, faces = np.asarray(V, dtype=float), np.asarray(F, dtype=int)
    result = {"status": "failed", "reason": "", "rmse": None, "coverage": 0.0}
    if len(vertices) < 4 or len(faces) < 4:
        result["reason"] = "empty mesh"
        return result
    try:
        import trimesh

        mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
        if not mesh.is_watertight:
            result["reason"] = "not watertight"
            return result
        unique_faces = np.unique(np.sort(faces, axis=1), axis=0)
        if len(unique_faces) != len(faces):
            result["reason"] = "duplicate faces"
            return result
        if not mesh.is_volume or mesh.volume <= 1e-9:
            result["reason"] = "not a positive volume"
            return result
        normals = mesh.face_normals
    except Exception as exc:
        result["reason"] = f"mesh check failed: {exc}"
        return result
    roof_indices = np.flatnonzero(normals[:, 2] > 0.1)
    if not len(roof_indices):
        result["reason"] = "no roof faces"
        return result
    roof_faces = faces[roof_indices]
    projected = []
    for face in roof_faces:
        polygon = Polygon(vertices[face, :2])
        if polygon.area > 1e-10:
            projected.append(polygon)
    if projected and footprint.area > 1e-10:
        result["coverage"] = float(unary_union(projected).intersection(footprint).area / footprint.area)
    roof_points = np.asarray(roof_points_xyz, dtype=float)
    if len(roof_points) and vertical:
        result["rmse"] = roof_height_rmse(vertices, roof_faces, roof_points, footprint, exclude=exclude)
        result["rmse_all"] = roof_height_rmse(vertices, roof_faces, roof_points, footprint)
    elif len(roof_points):
        try:
            roof_mesh = trimesh.Trimesh(vertices=vertices, faces=roof_faces, process=False)
            _closest, distances, _triangles = trimesh.proximity.closest_point(roof_mesh, roof_points)
        except Exception:
            distances = _fallback_distances(roof_points, vertices, roof_faces)
        result["rmse"] = float(np.sqrt(np.mean(np.asarray(distances, dtype=float) ** 2)))
    result.update(status="ok", reason="")
    return result
