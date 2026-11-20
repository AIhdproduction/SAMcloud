"""Opening evidence on arbitrary vertical planes.

The indoor refinement consumes plan-bound photo evidence.  Outdoor facades have no
BuildingPlan or camera-depth buffers, so this module keeps the same 5 cm grid idea
while using the labelled multi-view point cloud directly.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.ndimage as ndi
from shapely.geometry import Polygon

from samcloud.drone.contracts import DOOR, WINDOW


@dataclass(frozen=True)
class PlaneWallLine:
    """Small WallLine-compatible adapter for a vertical world-space plane."""

    id: str
    pieces: tuple[str, ...]
    a: np.ndarray
    d: np.ndarray
    n: np.ndarray
    length: float
    length_m: float
    thickness: float
    kind: str
    pad_start_m: float = 0.0
    pad_end_m: float = 0.0
    z_min: float = 0.0
    z_max: float = 0.0


def plane_wall_line(origin_xy, direction_xy, z_min, z_max, thickness=0.3) -> PlaneWallLine:
    """Build a minimal wall-line adapter for a vertical plane."""
    direction = np.asarray(direction_xy, dtype=float)
    direction /= np.linalg.norm(direction)
    normal = np.array([-direction[1], direction[0]], dtype=float)
    return PlaneWallLine(
        id="plane", pieces=("plane",), a=np.asarray(origin_xy, dtype=float), d=direction, n=normal,
        length=0.0, length_m=0.0, thickness=float(thickness), kind="exterior",
        z_min=float(z_min), z_max=float(z_max),
    )


def _geometry(plane):
    if hasattr(plane, "origin_xy"):
        origin = np.asarray(plane.origin_xy, dtype=float)
        direction = np.asarray(plane.direction_xy, dtype=float)
        length = float(plane.length)
        z_min, z_max = float(plane.z_min), float(plane.z_max)
        polygon = plane.polygon
    else:
        origin = np.asarray(plane.a, dtype=float)
        direction = np.asarray(plane.d, dtype=float)
        length = float(plane.length_m)
        z_min, z_max = float(plane.z_min), float(plane.z_max)
        polygon = None
    direction /= np.linalg.norm(direction)
    return origin, direction, length, z_min, z_max, polygon


def _candidate_components(mask: np.ndarray, weights: np.ndarray, counts: np.ndarray, views: np.ndarray, *, kind: str, cell: float, u_offset: float, z_offset: float, polygon) -> list[dict]:
    labels, number = ndi.label(mask)
    result: list[dict] = []
    for component_id in range(1, number + 1):
        component = labels == component_id
        rows, cols = np.nonzero(component)
        if not len(rows):
            continue
        r0, r1 = int(rows.min()), int(rows.max()) + 1
        c0, c1 = int(cols.min()), int(cols.max()) + 1
        u0, u1 = u_offset + c0 * cell, u_offset + c1 * cell
        z0, z1 = z_offset + r0 * cell, z_offset + r1 * cell
        rectangle = np.array([[u0, z0], [u1, z0], [u1, z1], [u0, z1]], dtype=float)
        inside = True if polygon is None else bool(polygon.covers(Polygon(rectangle)))
        support = component
        support_count = int(counts[support].sum())
        support_weight = float(weights[support].sum())
        if support_count == 0:
            continue
        area_cells = max(1, (r1 - r0) * (c1 - c0))
        result.append({
            "kind": kind, "u0": float(u0), "u1": float(u1), "z0": float(z0), "z1": float(z1),
            "points": support_count, "views": int(np.max(views[support])),
            "median_views": float(np.median(views[support])), "coverage": float(component.sum() / area_cells),
            "score": support_weight, "inside": inside,
        })
    return result


def detect_openings_on_plane(points_xyz, labels, view_count, plane, *, config) -> list[dict]:
    """Find labelled window and door rectangles in a facade-aligned 5 cm vote grid."""
    points = np.asarray(points_xyz, dtype=float)
    classes = np.asarray(labels)
    views = np.ones(len(points), dtype=float) if view_count is None else np.maximum(1, np.asarray(view_count, dtype=float))
    origin, direction, length, z_min, z_max, polygon = _geometry(plane)
    if len(points) == 0 or length <= 0.0 or z_max <= z_min:
        return []
    normal = np.array([-direction[1], direction[0]], dtype=float)
    relative = points[:, :2] - origin
    u = relative @ direction
    depth = relative @ normal
    cell = 0.05
    # Keep a small apron around the mesh wall.  It lets the caller reject a
    # component that straddles the facade boundary instead of silently clipping it.
    u_offset = -0.25
    u_end = length + 0.25
    nu = max(1, int(np.ceil((u_end - u_offset) / cell)))
    nv = max(1, int(np.ceil((z_max - z_min) / cell)))
    base = (np.abs(depth) <= 0.25) & (u >= u_offset) & (u < u_end) & (points[:, 2] >= z_min) & (points[:, 2] < z_max)
    if not base.any():
        return []
    columns = np.clip(((u[base] - u_offset) / cell).astype(int), 0, nu - 1)
    rows = np.clip(((points[base, 2] - z_min) / cell).astype(int), 0, nv - 1)
    flat = rows * nu + columns
    chosen_classes, chosen_views = classes[base], views[base]
    all_counts = np.bincount(flat, minlength=nv * nu).reshape(nv, nu)
    all_views = np.zeros(nv * nu, dtype=float)
    np.maximum.at(all_views, flat, chosen_views)
    candidates: list[dict] = []
    for code, kind in ((WINDOW, "window"), (DOOR, "door")):
        selected = chosen_classes == code
        if not selected.any():
            continue
        weighted = np.bincount(flat[selected], weights=chosen_views[selected], minlength=nv * nu).reshape(nv, nu)
        counts = np.bincount(flat[selected], minlength=nv * nu).reshape(nv, nu)
        mask = weighted > 0.0
        mask = ndi.binary_closing(mask, structure=np.ones((3, 3)))
        mask = ndi.binary_opening(mask, structure=np.ones((2, 2)))
        candidates.extend(_candidate_components(mask, weighted, counts, all_views.reshape(nv, nu), kind=kind, cell=cell, u_offset=u_offset, z_offset=z_min, polygon=polygon))
    return candidates
