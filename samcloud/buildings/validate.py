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


def validate_model(V: np.ndarray, F: np.ndarray, roof_points_xyz: np.ndarray, footprint: Polygon) -> dict:
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
    if len(roof_points):
        try:
            roof_mesh = trimesh.Trimesh(vertices=vertices, faces=roof_faces, process=False)
            _closest, distances, _triangles = trimesh.proximity.closest_point(roof_mesh, roof_points)
        except Exception:
            distances = _fallback_distances(roof_points, vertices, roof_faces)
        result["rmse"] = float(np.sqrt(np.mean(np.asarray(distances, dtype=float) ** 2)))
    result.update(status="ok", reason="")
    return result
