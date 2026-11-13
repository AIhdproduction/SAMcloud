"""Remove depth-supported phantom geometry that can only be seen through solid walls."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np

from samcloud.colmap.plane_fill import View, read_colmap_array


DEFAULTS = {
    "enabled": False, "keep_depth_m": 1.0, "cell_m": 0.05, "floor_band_m": 0.15,
    "extent_margin_m": 0.25, "blocked_share": 0.9, "min_support_views": 2,
    "mirror_image": {"enabled": False, "search_front_m": 1.2, "search_back_m": 0.2,
                     "step_m": 0.01, "match_m": 0.04, "min_score": 0.35, "min_points": 2000},
}


@dataclass
class WallGrid:
    index: int
    normal: np.ndarray
    offset: float
    tangent: np.ndarray
    u0: float
    z0: float
    cell: float
    solid: np.ndarray
    extent: tuple[float, float, float, float]
    solid_before: float = 0.0
    solid_after: float = 0.0


@dataclass
class FurnitureOccluder:
    """Tall, semantically identified furniture represented as a vertical oriented box."""

    label: str
    center: np.ndarray
    axes: np.ndarray
    half_size: np.ndarray

    @property
    def size(self) -> np.ndarray:
        return self.half_size * 2.0


OCCLUDER_FINE_TERMS = {"wardrobe", "bookcase", "shelf", "kitchen cabinet", "refrigerator"}


def _metric(settings: dict[str, Any], name: str) -> float:
    """Convert a configured metre value to the current reconstruction units."""
    return float(settings[name]) * float(settings.get("units_per_metre", 1.0))


def _wall_planes(plane_report: dict[str, Any]) -> list[dict[str, Any]]:
    return list((plane_report.get("wall") or {}).get("walls") or [])


def _mirror_cells(grid: WallGrid, mirrors: list[Any]) -> np.ndarray:
    mask = np.zeros_like(grid.solid, dtype=np.uint8)
    for mirror in mirrors:
        normal = np.asarray(getattr(mirror, "normal", ()), dtype=float)
        if normal.shape != (3,) or abs(float(normal @ grid.normal)) < 0.995:
            continue
        centre = np.asarray(getattr(mirror, "center", ()), dtype=float)
        if centre.shape != (3,) or abs(float(centre @ grid.normal + grid.offset)) > grid.cell * 2:
            continue
        # Mirror polygons live in the mirror's in-plane basis.  Reconstruct their
        # world coordinates before rasterising into this wall's along/height grid.
        first = np.cross(normal, np.array([1.0, 0.0, 0.0]) if abs(normal[0]) < 0.9 else np.array([0.0, 1.0, 0.0]))
        first /= max(np.linalg.norm(first), 1e-12)
        second = np.cross(normal, first)
        for polygon in getattr(mirror, "polygons", ()):  # confirmed mirrors are always opaque here
            polygon = np.asarray(polygon, dtype=float)
            if polygon.ndim != 2 or len(polygon) < 3:
                continue
            world = centre + polygon[:, :1] * first + polygon[:, 1:2] * second
            pixels = np.column_stack(((world @ grid.tangent - grid.u0) / grid.cell,
                                      (world[:, 2] - grid.z0) / grid.cell)).round().astype(np.int32)
            cv2.fillPoly(mask, [pixels], 1)
    return mask.astype(bool)


def build_wall_grids(points: np.ndarray, labels: np.ndarray, wall_index: np.ndarray,
                     plane_report: dict[str, Any], settings: dict[str, Any], class_ids: dict[str, int],
                     mirrors: list[Any] | None = None, confirmed_mirror_planes: set[int] | None = None) -> list[WallGrid]:
    """Build closed, semantic-aware solid/open occupancy grids for fitted walls."""
    cell = _metric(settings, "cell_m")
    margin = _metric(settings, "extent_margin_m")
    floor_band = _metric(settings, "floor_band_m")
    snap = _metric(settings, "plane_snap_m") if "plane_snap_m" in settings else cell
    wall_id, door_id, window_id, floor_id = (class_ids.get("wall"), class_ids.get("door"),
                                              class_ids.get("window"), class_ids.get("floor"))
    confirmed_mirror_planes = confirmed_mirror_planes or set()
    grids: list[WallGrid] = []
    for index, plane in enumerate(_wall_planes(plane_report)):
        normal = np.asarray(plane.get("normal", ()), dtype=float)
        if normal.shape != (3,) or np.linalg.norm(normal) < 1e-9:
            continue
        normal /= np.linalg.norm(normal)
        offset = float(plane.get("offset", 0.0))
        # Decimation can drop the assigned rows.  Include all points of this
        # plane's own wall class within the same snap tolerance used by fitting.
        rows = (wall_index == index) | ((labels == wall_id) & (np.abs(points @ normal + offset) <= snap))
        if wall_id is not None:
            rows &= labels == wall_id
        wall = points[rows]
        if len(wall) < 4:
            continue
        tangent = np.array((-normal[1], normal[0], 0.0))
        tangent /= max(np.linalg.norm(tangent), 1e-12)
        along = wall @ tangent
        extent = plane.get("extent")
        lo = float(extent[0] - margin) if extent is not None else float(np.min(along) - margin)
        hi = float(extent[1] + margin) if extent is not None else float(np.max(along) + margin)
        z0, z1 = float(np.min(wall[:, 2]) - margin), float(np.max(wall[:, 2]) + margin)
        width, height = max(1, int(np.ceil((hi - lo) / cell)) + 1), max(1, int(np.ceil((z1 - z0) / cell)) + 1)
        occupied = np.zeros((height, width), dtype=np.uint8)
        columns = np.clip(((along - lo) / cell).astype(int), 0, width - 1)
        lines = np.clip(((wall[:, 2] - z0) / cell).astype(int), 0, height - 1)
        occupied[lines, columns] = 1
        spacing = np.inf
        if len(wall) > 1:
            from scipy.spatial import cKDTree
            sample = wall if len(wall) <= 50_000 else wall[np.linspace(0, len(wall) - 1, 50_000, dtype=int)]
            spacing = float(np.median(cKDTree(sample).query(sample, k=2)[0][:, 1]))
        close_cells = 2 if spacing > 0.5 * cell else 1
        closed = cv2.morphologyEx(occupied, cv2.MORPH_CLOSE,
                                  np.ones((2 * close_cells + 1, 2 * close_cells + 1), np.uint8)).astype(bool)
        grid = WallGrid(index, normal, offset, tangent, lo, z0, cell, closed, (lo, hi, z0, z1),
                        float(closed.mean()), 0.0)
        opening = np.zeros_like(closed)
        near = np.abs(points @ normal + offset) <= cell * 1.5
        if door_id is not None or window_id is not None:
            opening_points = near & ((labels == door_id) if door_id is not None else False)
            if window_id is not None:
                opening_points |= near & (labels == window_id)
            q = points[opening_points]
            if len(q):
                x = np.clip(((q @ tangent - lo) / cell).astype(int), 0, width - 1)
                y = np.clip(((q[:, 2] - z0) / cell).astype(int), 0, height - 1)
                opening[y, x] = True
        # Floor-touching holes need independent door evidence.  A plain scan
        # gap at the floor is solid; mirrors frequently look exactly like that.
        empty = (~closed).astype(np.uint8)
        count, components = cv2.connectedComponents(empty)
        floor_rows = max(1, int(np.ceil(floor_band / cell)))
        door_components = np.unique(components[:floor_rows][components[:floor_rows] > 0])
        border_components = np.unique(np.concatenate((components[0], components[-1], components[:, 0], components[:, -1])))
        semantic_open = cv2.dilate(opening.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
        for component in range(1, count):
            member = components == component
            if component not in door_components and component not in border_components:
                closed[member] = True
                continue
            if component not in door_components:
                continue
            # The outside padding and a real door are one connected empty
            # component.  Split it at the measured wall-floor row so only an
            # interior gap, never the entire border component, can be opened.
            floor_line = int(np.clip(round((float(np.min(wall[:, 2])) - z0) / cell), 0, height - 1))
            columns_on_floor = np.flatnonzero(member[floor_line])
            spans = []
            for run in np.split(columns_on_floor, np.flatnonzero(np.diff(columns_on_floor) > 1) + 1):
                if len(run) and run[0] > 0 and run[-1] < width - 1:
                    spans.append((int(run[0]), int(run[-1])))
            for first, last in spans:
                hole = member.copy()
                hole[:, :first] = False
                hole[:, last + 1:] = False
                width_m = float(last - first + 1) * cell
                door_evidence = bool((semantic_open & hole).any())
                floor_continues = False
                floor = points[:, 2] <= z0 + margin + floor_band * 2
                if floor_id is not None and np.any(labels == floor_id):
                    floor &= labels == floor_id
                elif wall_id is not None:
                    floor &= labels != wall_id
                if np.any(floor):
                    along_floor = points[floor] @ tangent
                    low = lo + (first - 1) * cell
                    high = lo + (last + 2) * cell
                    floor_continues = int(np.count_nonzero((along_floor >= low) & (along_floor <= high))) >= max(4, (last - first + 1) // 2)
                metric = float(settings.get("units_per_metre", 1.0))
                natural_door = (0.6 * metric <= width_m <= 1.4 * metric and floor_continues and
                                index not in confirmed_mirror_planes)
                if door_evidence or natural_door:
                    opening |= hole
                else:
                    closed[hole] = True
        opening = cv2.dilate(opening.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
        grid.solid = (closed | _mirror_cells(grid, mirrors or [])) & ~opening
        grid.solid_after = float(grid.solid.mean())
        grids.append(grid)
    return grids


def _solid_at(grid: WallGrid, points: np.ndarray) -> np.ndarray:
    x = ((points @ grid.tangent - grid.u0) / grid.cell).astype(int)
    y = ((points[:, 2] - grid.z0) / grid.cell).astype(int)
    inside = (x >= 0) & (x < grid.solid.shape[1]) & (y >= 0) & (y < grid.solid.shape[0])
    out = np.zeros(len(points), dtype=bool)
    out[inside] = grid.solid[y[inside], x[inside]]
    return out


def _voxel_components(points: np.ndarray, cell: float) -> list[np.ndarray]:
    """Return 26-connected point components in a 3-D voxel grid."""
    if not len(points):
        return []
    voxels = np.floor(points / cell).astype(np.int64)
    members: dict[tuple[int, int, int], list[int]] = {}
    for row, voxel in enumerate(voxels):
        members.setdefault(tuple(int(value) for value in voxel), []).append(row)
    unseen = set(members)
    components: list[np.ndarray] = []
    neighbours = [(dx, dy, dz) for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1)]
    while unseen:
        seed = unseen.pop()
        queue = [seed]
        rows: list[int] = []
        while queue:
            voxel = queue.pop()
            rows.extend(members[voxel])
            for dx, dy, dz in neighbours:
                nearby = (voxel[0] + dx, voxel[1] + dy, voxel[2] + dz)
                if nearby in unseen:
                    unseen.remove(nearby)
                    queue.append(nearby)
        components.append(np.asarray(rows, dtype=np.int64))
    return components


def build_furniture_occluders(points: np.ndarray, labels: np.ndarray, fine_terms: np.ndarray | None,
                              fine_labels: list[str] | None, settings: dict[str, Any],
                              class_ids: dict[str, int]) -> list[FurnitureOccluder]:
    """Build protected solid boxes from tall SAM fine-term furniture clusters."""
    if fine_terms is None or fine_labels is None or len(fine_terms) != len(points):
        return []
    floor_id = class_ids.get("floor")
    floor_points = points[labels == floor_id] if floor_id is not None else np.empty((0, 3))
    if not len(floor_points):
        return []
    floor_z = float(np.quantile(floor_points[:, 2], 0.05))
    cell = _metric(settings, "cell_m")
    extension = cell
    min_height = 1.4 * float(settings.get("units_per_metre", 1.0))
    boxes: list[FurnitureOccluder] = []
    term_values = np.asarray(fine_terms, dtype=np.int64)
    for value, label in enumerate(fine_labels, start=1):
        label = str(label).strip().lower()
        if label not in OCCLUDER_FINE_TERMS:
            continue
        rows = np.flatnonzero(term_values == value)
        for component in _voxel_components(points[rows], cell):
            members = rows[component]
            cluster = points[members]
            top = float(np.max(cluster[:, 2]))
            if len(cluster) < 4 or top - floor_z < min_height:
                continue
            rectangle = cv2.minAreaRect(cluster[:, :2].astype(np.float32))
            (cx, cy), (width, depth), angle = rectangle
            if width <= 0 or depth <= 0:
                continue
            theta = np.deg2rad(float(angle))
            major = np.array((np.cos(theta), np.sin(theta), 0.0))
            minor = np.array((-np.sin(theta), np.cos(theta), 0.0))
            axes = np.vstack((major, minor, np.array((0.0, 0.0, 1.0))))
            half_size = np.array((width / 2.0 + extension, depth / 2.0 + extension,
                                  (top - floor_z) / 2.0 + extension))
            center = np.array((cx, cy, (floor_z + top) / 2.0))
            boxes.append(FurnitureOccluder(label, center, axes, half_size))
    return boxes


def _inside_furniture(boxes: list[FurnitureOccluder], points: np.ndarray) -> np.ndarray:
    inside = np.zeros(len(points), dtype=bool)
    for box in boxes:
        local = (points - box.center) @ box.axes.T
        inside |= np.all(np.abs(local) <= box.half_size + 1e-9, axis=1)
    return inside


def _furniture_crossed(boxes: list[FurnitureOccluder], camera: np.ndarray, points: np.ndarray,
                       margin: float) -> np.ndarray:
    """Whether each camera-to-point segment intersects a solid furniture box."""
    crossed = np.zeros(len(points), dtype=bool)
    if not len(points):
        return crossed
    direction = points - camera
    length = np.linalg.norm(direction, axis=1)
    for box in boxes:
        origin = (camera - box.center) @ box.axes.T
        ray = direction @ box.axes.T
        lower, upper = -box.half_size, box.half_size
        with np.errstate(divide="ignore", invalid="ignore"):
            first = (lower - origin) / ray
            last = (upper - origin) / ray
        near = np.maximum.reduce(np.minimum(first, last), axis=1)
        far = np.minimum.reduce(np.maximum(first, last), axis=1)
        parallel_outside = (np.abs(ray) < 1e-12) & ((origin < lower) | (origin > upper))
        hit = ~parallel_outside.any(axis=1) & (far >= near) & (far > 0.0) & (near < 1.0)
        entry = np.maximum(near, 0.0) * length
        exit_ = np.minimum(far, 1.0) * length
        crossed |= hit & (exit_ - entry > 0.0) & (entry > margin) & ((1.0 - np.minimum(far, 1.0)) * length > margin)
    return crossed


def _camera_side(grid: WallGrid, views: list[View]) -> float:
    """The room side is the median camera side of a fitted wall."""
    signed = np.asarray([np.asarray(view.center) @ grid.normal + grid.offset for view in views], dtype=float)
    return 1.0 if float(np.median(signed)) >= 0 else -1.0


def _mirror_candidates(points: np.ndarray, grid: WallGrid, side: float, margin: float,
                       maximum: float) -> np.ndarray:
    signed = points @ grid.normal + grid.offset
    along = points @ grid.tangent
    lo, hi, z0, z1 = grid.extent
    return np.flatnonzero((side * signed < -margin) & (-side * signed <= maximum) &
                          (along >= lo) & (along <= hi) & (points[:, 2] >= z0) & (points[:, 2] <= z1))


def _match_reflections(tree: Any, points: np.ndarray, candidates: np.ndarray, normal: np.ndarray,
                       offset: float, side: float, match: float, clearance: float) -> np.ndarray:
    """Match reflections against room-side geometry using one global KD-tree."""
    far = side * (points[candidates] @ normal + offset) <= -clearance
    matched = np.zeros(len(candidates), dtype=bool)
    if not np.any(far):
        return matched
    reflected = points[candidates[far]] - 2.0 * (points[candidates[far]] @ normal + offset)[:, None] * normal
    _distances, neighbours = tree.query(reflected, distance_upper_bound=match)
    valid = neighbours < len(points)
    room_side = np.zeros(len(neighbours), dtype=bool)
    room_side[valid] = side * (points[neighbours[valid]] @ normal + offset) >= clearance
    matched[np.flatnonzero(far)] = valid & room_side
    return matched


def _connected_nearby(points: np.ndarray, candidates: np.ndarray, matched: np.ndarray, cell: float) -> np.ndarray:
    """Include occupied 0.05 m voxels within 0.10 m of a reflected match."""
    result = matched.copy()
    if not np.any(matched):
        return result
    from scipy.spatial import cKDTree
    voxels = np.floor(points[candidates] / cell).astype(np.int64)
    seed_voxels = np.unique(voxels[matched], axis=0)
    result |= cKDTree(seed_voxels.astype(float)).query(voxels.astype(float), p=np.inf)[0] <= 2.0
    return result


def _confirm_mirror_image(tree: Any, points: np.ndarray, grid: WallGrid, views: list[View],
                          settings: dict[str, Any], offsets: np.ndarray, source: str) -> tuple[np.ndarray, dict[str, Any] | None]:
    image = settings["mirror_image"]
    side = _camera_side(grid, views)
    margin = float(settings.get("visibility_margin_m", 0.03)) * float(settings.get("units_per_metre", 1.0))
    candidates = _mirror_candidates(points, grid, side, margin, _metric(settings, "mirror_search_max_m"))
    if len(candidates) < int(image["min_points"]):
        return np.zeros(len(points), dtype=bool), None
    sample = candidates if len(candidates) <= 10_000 else candidates[np.linspace(0, len(candidates) - 1, 10_000, dtype=int)]
    match = float(image["match_m"]) * float(settings.get("units_per_metre", 1.0))
    clearance = 0.30 * float(settings.get("units_per_metre", 1.0))
    scores = []
    for distance in offsets:
        scores.append(float(_match_reflections(tree, points, sample, grid.normal,
                                                grid.offset - side * float(distance), side, match, clearance).mean()))
    best = int(np.argmax(scores))
    distance, score = float(offsets[best]), float(scores[best])
    full = _match_reflections(tree, points, candidates, grid.normal, grid.offset - side * distance, side, match, clearance)
    median = float(np.median(scores))
    matched = int(full.sum())
    if score < float(image["min_score"]) or score < 2.0 * median or matched < int(image["min_points"]):
        return np.zeros(len(points), dtype=bool), None
    removed = np.zeros(len(points), dtype=bool)
    removed[candidates[_connected_nearby(points, candidates, full, _metric(settings, "cell_m"))]] = True
    return removed, {"wall_index": grid.index, "offset_d": round(distance, 4), "score": round(score, 4),
                     "matched_points": matched, "removed_points": int(removed.sum()), "source": source,
                     "plane_offset": round(grid.offset - side * distance, 4)}


def _mirror_image_mask(points: np.ndarray, grids: list[WallGrid], views: list[View], settings: dict[str, Any],
                       mirrors: list[Any] | None) -> tuple[np.ndarray, list[dict[str, Any]], set[int]]:
    image = settings.get("mirror_image") or {}
    if not bool(image.get("enabled", False)):
        return np.zeros(len(points), dtype=bool), [], set()
    from scipy.spatial import cKDTree
    tree = cKDTree(points)
    front, back, step = (_metric(settings, "mirror_front_m"), _metric(settings, "mirror_back_m"),
                         _metric(settings, "mirror_step_m"))
    offsets = np.arange(front, -back - step * 0.5, -step)
    # A plane essentially coincident with a wall simply reflects wall points
    # onto themselves.  It is never evidence of a real mirror.
    offsets = offsets[np.abs(offsets) > 0.10 * float(settings.get("units_per_metre", 1.0))]
    removed = np.zeros(len(points), dtype=bool)
    reports: list[dict[str, Any]] = []
    confirmed: set[int] = set()
    for grid in grids:
        found, record = _confirm_mirror_image(tree, points, grid, views, settings, offsets, "wall_search")
        if record is not None:
            removed |= found
            reports.append(record)
            confirmed.add(grid.index)
    for mirror in mirrors or []:
        normal = np.asarray(getattr(mirror, "normal", ()), dtype=float)
        if normal.shape != (3,) or np.linalg.norm(normal) < 1e-9:
            continue
        normal /= np.linalg.norm(normal)
        mirror_offset = float(getattr(mirror, "offset", np.nan))
        if not np.isfinite(mirror_offset):
            continue
        for grid in grids:
            if abs(float(normal @ grid.normal)) < 0.995:
                continue
            aligned = mirror_offset if normal @ grid.normal >= 0 else -mirror_offset
            distance = _camera_side(grid, views) * (grid.offset - aligned)
            if abs(distance) <= 0.10 * float(settings.get("units_per_metre", 1.0)):
                continue
            found, record = _confirm_mirror_image(tree, points, grid, views, settings,
                                                  np.asarray([distance]), "sam_mirror")
            if record is not None:
                removed |= found
                reports.append(record)
                confirmed.add(grid.index)
    return removed, reports, confirmed


def behind_wall_mask(points: np.ndarray, labels: np.ndarray, wall_index: np.ndarray,
                     plane_report: dict[str, Any], views: list[View], stereo_dir: Path,
                     settings: dict[str, Any], *, mirrors: list[Any] | None = None,
                     class_ids: dict[str, int] | None = None, depth_reader: Callable[[Path], np.ndarray] = read_colmap_array,
                     seen: np.ndarray | None = None, fine_terms: np.ndarray | None = None,
                     fine_labels: list[str] | None = None, device: str = "cpu",
                     chunk: int = 300_000) -> tuple[np.ndarray, dict[str, Any]]:
    """Return a keep mask for points whose depth-supported rays cross solid walls."""
    active = {**settings, **DEFAULTS, **(settings.get("behind_wall_filter") or {})}
    active["mirror_image"] = {**DEFAULTS["mirror_image"], **active.get("mirror_image", {})}
    active["mirror_front_m"] = active["mirror_image"]["search_front_m"]
    active["mirror_back_m"] = active["mirror_image"]["search_back_m"]
    active["mirror_step_m"] = active["mirror_image"]["step_m"]
    active["mirror_search_max_m"] = 4.0
    keep = np.ones(len(points), dtype=bool)
    empty_stats = {"behind_wall_removed": 0, "behind_wall_planes": [], "mirror_image_planes": [],
                   "mirror_image_removed": 0, "furniture_occluders": {"count": 0, "boxes": []}}
    if not bool(active["enabled"]) or not len(points) or not views:
        return keep, empty_stats
    ids = class_ids or {"wall": 2, "floor": 4, "door": 5, "window": 6}
    grids = build_wall_grids(points, labels, wall_index, plane_report, active, ids, mirrors)
    if seen is not None:
        minimum_seen = float(active.get("wall_min_seen_views", 3))
        grids = [grid for grid in grids if np.any(wall_index == grid.index) and
                 float(np.median(np.asarray(seen)[wall_index == grid.index])) >= minimum_seen]
    grids = [grid for grid in grids if bool(grid.solid.any())]
    if not grids:
        return keep, empty_stats
    furniture = build_furniture_occluders(points, labels, fine_terms, fine_labels, active, ids)
    furniture_inside = _inside_furniture(furniture, points)
    mirror_removed, mirror_reports, confirmed_mirrors = _mirror_image_mask(points, grids, views, active, mirrors)
    if confirmed_mirrors:
        grids = build_wall_grids(points, labels, wall_index, plane_report, active, ids, mirrors, confirmed_mirrors)
        grids = [grid for grid in grids if bool(grid.solid.any())]
    keep_depth = _metric(active, "keep_depth_m")
    grid_sides = {grid.index: _camera_side(grid, views) for grid in grids}
    margin_abs = float(active.get("visibility_margin_m", 0.03)) * float(active.get("units_per_metre", 1.0))
    margin_rel = float(active.get("visibility_margin_rel", 0.01))
    max_depth = float(active.get("max_geometric_depth", np.inf))
    wall_candidate = np.zeros(len(points), dtype=bool)
    for grid in grids:
        signed = points @ grid.normal + grid.offset
        along = points @ grid.tangent
        lo, hi, z0, z1 = grid.extent
        side = grid_sides[grid.index]
        wall_candidate |= (side * signed < -keep_depth) & (along >= lo) & (along <= hi) & (points[:, 2] >= z0) & (points[:, 2] <= z1)
    # Furniture augments the solid crossings for the same behind-wall
    # candidates; it never bypasses the configured built-in-depth allowance.
    furniture_candidate = wall_candidate & ~furniture_inside
    candidate = wall_candidate
    candidates = np.flatnonzero(candidate & ~mirror_removed)
    support = np.zeros(len(candidates), np.int16)
    blocked = np.zeros(len(candidates), np.int16)
    furniture_blocked = np.zeros(len(candidates), np.int16)
    direct = np.zeros(len(candidates), np.int16)
    if len(candidates):
        # Depth agreement mirrors visibility_mask exactly; only those views may
        # vote for a blocked ray.  Projection is torch/GPU chunked.
        import torch
        p = torch.as_tensor(points[candidates], dtype=torch.float32, device=device)
        for view in views:
            path = Path(stereo_dir) / "depth_maps" / f"{view.name}.geometric.bin"
            if not path.is_file():
                continue
            depth = np.asarray(depth_reader(path), dtype=np.float32)
            depth[(depth <= 0) | (depth > max_depth)] = 0
            d = torch.as_tensor(depth, dtype=torch.float32, device=device)
            rotation = torch.as_tensor(view.rotation, dtype=torch.float32, device=device)
            translation = torch.as_tensor(view.translation, dtype=torch.float32, device=device)
            camera = np.asarray(view.center, dtype=float)
            for start in range(0, len(candidates), chunk):
                part = p[start:start + chunk]
                local = part @ rotation.T + translation
                z = local[:, 2]
                u = (view.fx * local[:, 0] / z + view.cx) * d.shape[1] / view.width
                v = (view.fy * local[:, 1] / z + view.cy) * d.shape[0] / view.height
                valid = (z > 0.1) & (u >= 0) & (u < d.shape[1]) & (v >= 0) & (v < d.shape[0])
                pos = torch.nonzero(valid, as_tuple=False).squeeze(1)
                if not len(pos):
                    continue
                measured = d[v[pos].long(), u[pos].long()]
                ok = (measured > 0) & ((measured - z[pos]).abs() <= margin_abs + margin_rel * z[pos])
                if not bool(ok.any()):
                    continue
                rows = pos[ok].cpu().numpy()
                world = part[pos[ok]].cpu().numpy()
                support[start + rows] += 1
                ray = world - camera
                crossed = np.zeros(len(world), dtype=bool)
                directly_seen = np.zeros(len(world), dtype=bool)
                ray_length = np.linalg.norm(ray, axis=1)
                for grid in grids:
                    point_side = world @ grid.normal + grid.offset
                    camera_side = float(camera @ grid.normal + grid.offset)
                    expected_back = grid_sides[grid.index] * point_side < -keep_depth
                    directly_seen |= expected_back & (point_side * camera_side > margin_abs**2)
                    denominator = ray @ grid.normal
                    fraction = -(camera @ grid.normal + grid.offset) / np.where(np.abs(denominator) > 1e-10, denominator, np.inf)
                    valid_crossing = (fraction > 0) & (fraction < 1)
                    valid_crossing &= (fraction * ray_length > margin_abs) & ((1 - fraction) * ray_length > margin_abs)
                    if valid_crossing.any():
                        hit = camera + ray[valid_crossing] * fraction[valid_crossing, None]
                        local_crossed = _solid_at(grid, hit)
                        crossed[np.flatnonzero(valid_crossing)[local_crossed]] = True
                crossed_by_furniture = _furniture_crossed(furniture, camera, world, margin_abs)
                crossed |= crossed_by_furniture
                blocked[start + rows[crossed]] += 1
                furniture_blocked[start + rows[crossed_by_furniture]] += 1
                direct[start + rows[directly_seen]] += 1
    minimum = int(active["min_support_views"])
    # A real neighbouring room can be confirmed by a camera standing on its
    # side of the wall.  Its depth support is stronger evidence than rays from
    # the other room that happen to cross a fitted wall, so never remove it.
    blocking_share = blocked.astype(float) >= float(active["blocked_share"]) * support
    wall_remove = wall_candidate[candidates] & (support >= minimum) & blocking_share & (direct == 0)
    furniture_remove = (furniture_candidate[candidates] & (support >= minimum) &
                        (furniture_blocked.astype(float) >= float(active["blocked_share"]) * support))
    remove = wall_remove | furniture_remove
    ray_removed = np.zeros(len(points), dtype=bool)
    ray_removed[candidates[remove]] = True
    removed = (ray_removed | mirror_removed) & ~furniture_inside
    keep[removed] = False
    # Attribute every removal to one nearest qualifying wall.  A point can be
    # behind several planes, but the report must not count it more than once.
    assignment = np.full(len(points), -1, np.int64)
    nearest = np.full(len(points), np.inf)
    for grid in grids:
        signed = points @ grid.normal + grid.offset
        along = points @ grid.tangent
        lo, hi, z0, z1 = grid.extent
        distance = -grid_sides[grid.index] * signed
        eligible = (distance > keep_depth) & (along >= lo) & (along <= hi) & (points[:, 2] >= z0) & (points[:, 2] <= z1)
        take = eligible & (distance < nearest)
        assignment[take], nearest[take] = grid.index, distance[take]
    per_plane = []
    for grid in grids:
        per_plane.append({"index": grid.index, "removed_points": int(np.count_nonzero(removed & (assignment == grid.index))),
                          "solid_share": round(grid.solid_after, 4), "solid_share_before": round(grid.solid_before, 4),
                          "solid_share_after": round(grid.solid_after, 4)})
    box_records = [{"label": box.label, "center": [round(float(value), 4) for value in box.center],
                    "size": [round(float(value), 4) for value in box.size]}
                   for box in furniture]
    return keep, {"behind_wall_removed": int(removed.sum()), "behind_wall_directly_seen_kept": int(np.count_nonzero(direct)),
                  "furniture_occluder_removed": int(np.count_nonzero(furniture_remove)),
                  "behind_wall_planes": per_plane,
                  "mirror_image_planes": mirror_reports, "mirror_image_removed": int((mirror_removed & ~furniture_inside).sum()),
                  "furniture_occluders": {"count": len(furniture), "boxes": box_records}}
