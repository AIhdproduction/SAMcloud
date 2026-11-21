"""Wall surfaces derived directly from an LoD2 building mesh."""

from __future__ import annotations

from dataclasses import dataclass
from math import acos, degrees

import numpy as np
from shapely.geometry import Polygon
from shapely.ops import unary_union

from samcloud.drone.contracts import BuildingModel, FacadePlane


GROUP_ANGLE_DEG = 3.0  # coplanar wall faces: normals within this angle
GROUP_OFFSET_M = 0.10  # ... and plane offsets within this distance


@dataclass(frozen=True)
class FacadeSurface:
    """One coplanar exterior wall in a mesh-aligned ``(u, z)`` frame."""

    building_id: int
    facade_index: int
    origin_xy: np.ndarray
    direction_xy: np.ndarray
    normal_xy: np.ndarray
    length: float
    z_min: float
    z_max: float
    polygon: Polygon
    face_indices: tuple[int, ...]
    facade_plane: FacadePlane | None = None


def _unit(vector: np.ndarray) -> np.ndarray:
    length = float(np.linalg.norm(vector))
    if length <= 1e-12:
        raise ValueError("zero-length facade direction")
    return vector / length


def _canonical_plane(vertices: np.ndarray) -> tuple[np.ndarray, float]:
    """Return a stable vertical plane normal and offset for a wall triangle."""
    cross = np.cross(vertices[1] - vertices[0], vertices[2] - vertices[0])
    if float(np.linalg.norm(cross)) <= 1e-10 or float(np.linalg.norm(cross[:2])) <= 1e-10:
        raise ValueError("degenerate wall triangle")
    normal = cross[:2]
    normal = _unit(normal)
    if normal[0] < -1e-10 or (abs(normal[0]) <= 1e-10 and normal[1] < 0.0):
        normal = -normal
    return normal, -float(normal @ vertices[0, :2])


def _match_plane(normal: np.ndarray, offset: float, facades: list[FacadePlane]) -> FacadePlane | None:
    best: tuple[float, FacadePlane] | None = None
    for facade in facades:
        candidate = np.asarray(facade.normal, dtype=float)[:2]
        if np.linalg.norm(candidate) <= 1e-12:
            continue
        candidate = _unit(candidate)
        dot = abs(float(normal @ candidate))
        angle = degrees(acos(float(np.clip(dot, -1.0, 1.0))))
        if angle >= 5.0:
            continue
        candidate_offset = float(facade.d)
        if float(normal @ candidate) < 0.0:
            candidate_offset = -candidate_offset
        distance = abs(offset - candidate_offset)
        if distance >= 0.3:
            continue
        score = angle + distance / 0.3
        if best is None or score < best[0]:
            best = (score, facade)
    return None if best is None else best[1]


def _direction(normal: np.ndarray, plane: FacadePlane | None) -> np.ndarray:
    if plane is not None and plane.segment is not None and plane.segment.length > 1e-9:
        coords = list(plane.segment.coords)
        direction = _unit(np.asarray(coords[-1], dtype=float)[:2] - np.asarray(coords[0], dtype=float)[:2])
        if abs(float(direction @ normal)) < np.sin(np.deg2rad(5.0)):
            return direction
    direction = np.array([-normal[1], normal[0]], dtype=float)
    if direction[0] < -1e-10 or (abs(direction[0]) <= 1e-10 and direction[1] < 0.0):
        direction = -direction
    return direction


def _group_wall_faces(model: BuildingModel, report: dict | None = None) -> list[dict]:
    vertices = np.asarray(model.vertices, dtype=float)
    faces = np.asarray(model.faces, dtype=int)
    roles = np.asarray(model.face_roles)
    groups: list[dict] = []
    for index in np.flatnonzero(roles == 1):
        triangle = vertices[faces[index]]
        try:
            normal, offset = _canonical_plane(triangle)
        except ValueError:
            if report is not None:
                report["degenerate_wall_triangles"] = int(report.get("degenerate_wall_triangles", 0)) + 1
            continue
        for group in groups:
            if degrees(acos(float(np.clip(abs(group["normal"] @ normal), -1.0, 1.0)))) >= GROUP_ANGLE_DEG:
                continue
            aligned_offset = offset if float(group["normal"] @ normal) >= 0 else -offset
            if abs(group["offset"] - aligned_offset) < GROUP_OFFSET_M:
                group["faces"].append(int(index))
                group["vertices"].append(triangle)
                raw = np.cross(triangle[1] - triangle[0], triangle[2] - triangle[0])[:2]
                if np.linalg.norm(raw) > 1e-12:
                    group["outward"].append(_unit(raw))
                break
        else:
            raw = np.cross(triangle[1] - triangle[0], triangle[2] - triangle[0])[:2]
            groups.append({"normal": normal, "offset": offset, "faces": [int(index)], "vertices": [triangle], "outward": [_unit(raw)]})
    return groups


def facade_surfaces(model: BuildingModel, facades: list[FacadePlane], *, report: dict | None = None) -> list[FacadeSurface]:
    """Return one mesh-exact surface for each coplanar LoD2 wall face group."""
    surfaces: list[FacadeSurface] = []
    if report is not None:
        report.setdefault("degenerate_wall_triangles", 0)
    for group in _group_wall_faces(model, report):
        vertices = np.concatenate(group["vertices"], axis=0)
        plane = _match_plane(group["normal"], group["offset"], facades)
        direction = _direction(group["normal"], plane)
        projected = vertices[:, :2] @ direction
        start = float(projected.min())
        origin = direction * start - group["normal"] * group["offset"]
        uv_triangles = []
        for triangle in group["vertices"]:
            uv_triangles.append(Polygon(np.column_stack(((triangle[:, :2] - origin) @ direction, triangle[:, 2]))))
        merged = unary_union(uv_triangles)
        if not isinstance(merged, Polygon):
            polygons = [item for item in getattr(merged, "geoms", []) if isinstance(item, Polygon)]
            if not polygons:
                continue
            merged = max(polygons, key=lambda item: item.area)
        mean_outward = np.mean(group["outward"], axis=0)
        # Opposite-facing coplanar roof-step walls can cancel; keep the first normal then.
        outward = _unit(mean_outward if float(np.linalg.norm(mean_outward)) > 1e-6 else group["outward"][0])
        surfaces.append(FacadeSurface(
            building_id=model.building_id,
            facade_index=len(surfaces),
            origin_xy=np.asarray(origin, dtype=float),
            direction_xy=direction,
            normal_xy=outward,
            length=float(projected.max() - start),
            z_min=float(vertices[:, 2].min()),
            z_max=float(vertices[:, 2].max()),
            polygon=merged,
            face_indices=tuple(group["faces"]),
            facade_plane=plane,
        ))
    return surfaces
