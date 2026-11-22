"""Open door leaves that the wall-face extraction fitted as walls (rule: a door is never a wall).

An open door leaf is a 4 cm thick, about 2 m high slab of door-class points.  The wall-face extraction streams every structural
class (wall, door, window, column) and therefore fits a "wall" along a leaf that stands open into the room; that stub closes the
doorway and the door disappears.  The face is often longer than the leaf: it runs on into the wall the leaf is hinged at.

Only the free-standing panel part of a face is removed, measured stretch by stretch (10 cm):

* a leaf stretch has its supporting points on door-class cells of the top-view class raster, and no measured point near it
  (within ``wall_search_m``, so also a wall a frame depth behind a leaf lying open against it) reaches clearly above the door head;
* the leaf part is the longest run of leaf stretches, a door-leaf length long (``min_leaf_m`` .. ``max_leaf_m``);
* the floor is free on BOTH sides of it (a leaf stands free in the room; a closed leaf in a wall has the room on one side only);
* it does not lie on the line of another (collinear) wall face, nor on that line's extension (a leaf in a wall gap is no stub).

The rest of the face (the stretch inside the wall the leaf is hinged at, a wall continuing past the door) is kept as its own face.
The raster cells of the removed panel are no wall evidence either (``leaf_face_mask``), clipped to stay clear of every kept wall
face and its extension, so the walls and rooms around the door never change.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import cv2
import numpy as np
from shapely.geometry import LineString

from .raster_grid import RasterGrid


@dataclass(frozen=True)
class DoorLeafFaceSettings:
    enabled: bool = True  # v12 pass 1 identical to baseline (nothing removed there), bench apartment_104 two-leaf corridor fixed (2026-11-22)
    min_door_share: float = 0.5  # share of a stretch's supporting points on door-class raster cells
    max_top_fraction: float = 0.88  # a leaf ends at the door head: no point near it above this share of the storey height
    high_points: int = 3  # ... (at most this many points above it per stretch count as noise)
    dilate_m: float = 0.03  # door-class raster dilation (the face line runs along the leaf surface, not its centre)
    wall_search_m: float = 0.30  # points this close to the face count for the height test (a wall a frame depth behind the leaf)
    stretch_m: float = 0.10
    min_leaf_m: float = 0.40  # the free-standing panel is a door-leaf length long ...
    max_leaf_m: float = 2.60  # ... (two leaves of neighbouring doors standing in line merge into one run)
    floor_probe_m: tuple[float, float] = (0.15, 0.45)  # free floor is probed this far on both sides of the panel ...
    floor_min_share: float = 0.5  # ... and must be found at this share of the probes on EACH side
    protect_m: float = 0.20  # erased cells and removed panels stay this far from every kept wall face ...
    protect_extension_m: float = 0.50  # ... and from its extension beyond its ends
    min_piece_m: float = 0.10  # a kept rest of a face shorter than this is dropped with the panel


def _support_on_mask(points: np.ndarray, mask: np.ndarray, grid: RasterGrid) -> np.ndarray:
    rows = np.floor((grid.max_y - points[:, 1]) / grid.pixel_size).astype(np.int64)
    cols = np.floor((points[:, 0] - grid.min_x) / grid.pixel_size).astype(np.int64)
    inside = (rows >= 0) & (rows < mask.shape[0]) & (cols >= 0) & (cols < mask.shape[1])
    hit = np.zeros(len(points), dtype=bool)
    hit[inside] = mask[rows[inside], cols[inside]]
    return hit


def _axis(line: LineString) -> tuple[np.ndarray, np.ndarray, float]:
    start, end = (np.asarray(line.coords[index], dtype=float) for index in (0, -1))
    length = float(np.linalg.norm(end - start))
    return start, (end - start) / max(length, 1e-12), length


def _leaf_stretches(face: Any, support: np.ndarray, door_hit: np.ndarray, wall_points: np.ndarray, settings: DoorLeafFaceSettings, upm: float, storey_height: float) -> np.ndarray:
    """Per 10 cm stretch of the face: True when it looks like a free-standing leaf (door-class support, nothing above the door head)."""
    start, axis, length = _axis(face.line)
    step = settings.stretch_m * upm
    count = max(1, int(np.ceil(length / step)))
    normal = np.array((-axis[1], axis[0]))
    s_along = (support - start) @ axis
    relative = wall_points[:, :2] - start
    w_along = relative @ axis
    w_near = (np.abs(relative @ normal) <= max(settings.dilate_m, settings.wall_search_m) * upm) & (w_along >= -step) & (w_along <= length + step)
    w_along, w_high = w_along[w_near], wall_points[w_near, 2] > settings.max_top_fraction * storey_height
    result = np.zeros(count, dtype=bool)
    for index in range(count):
        lo, hi = index * step, (index + 1) * step
        in_stretch = (s_along >= lo) & (s_along < hi)
        if np.count_nonzero(in_stretch) < 3 or float(door_hit[in_stretch].mean()) < settings.min_door_share:
            continue
        high = np.count_nonzero(w_high & (w_along >= lo) & (w_along < hi))
        result[index] = high < settings.high_points
    return result


def _longest_run(mask: np.ndarray, max_gap: int = 1) -> tuple[int, int] | None:
    index = np.flatnonzero(mask)
    if index.size == 0:
        return None
    best, start, previous = None, int(index[0]), int(index[0])
    for value in [*index[1:].tolist(), None]:
        if value is not None and value - previous <= max_gap + 1:
            previous = value
            continue
        if best is None or previous + 1 - start > best[1] - best[0]:
            best = (start, previous + 1)
        if value is not None:
            start = previous = value
    return best


def _free_both_sides(segment: tuple[np.ndarray, np.ndarray], free: np.ndarray | None, grid: RasterGrid, settings: DoorLeafFaceSettings, upm: float) -> bool:
    if free is None:
        return True
    a, b = segment
    axis = (b - a) / max(float(np.linalg.norm(b - a)), 1e-12)
    normal = np.array((-axis[1], axis[0]))
    near, far = (value * upm for value in settings.floor_probe_m)
    along = np.linspace(0.1, 0.9, 7)
    for sign in (1.0, -1.0):
        hits = []
        for t in along:
            for distance in np.linspace(near, far, 3):
                point = a + (b - a) * t + sign * normal * distance
                row, col = grid.local_to_pixel(float(point[0]), float(point[1]))
                row, col = int(round(row)), int(round(col))
                hits.append(0 <= row < free.shape[0] and 0 <= col < free.shape[1] and bool(free[row, col]))
        if float(np.mean(hits)) < settings.floor_min_share:
            return False
    return True


def _protect_mask(lines: list[LineString], grid: RasterGrid, upm: float, settings: DoorLeafFaceSettings, parallel_to: np.ndarray | None = None, unextended: list[LineString] = ()) -> np.ndarray:
    """Cells within ``protect_m`` of the given wall face lines extended by ``protect_extension_m`` (optionally only lines parallel to an axis).

    ``unextended`` lines are protected without the extension (the wall stretch of a leaf face itself must not shield its own panel).
    """
    mask = np.zeros((grid.height, grid.width), dtype=np.uint8)
    thickness = max(1, int(round(2.0 * settings.protect_m * upm / grid.pixel_size)))
    for line in [*lines, *unextended]:
        extension = 0.0 if any(line is other for other in unextended) else settings.protect_extension_m * upm
        start, axis, length = _axis(line)
        if length <= 1e-9:
            continue
        if parallel_to is not None and abs(float(axis[0] * parallel_to[1] - axis[1] * parallel_to[0])) > np.sin(np.radians(10.0)):
            continue
        a, b = start - axis * extension, start + axis * (length + extension)
        r0, c0 = grid.local_to_pixel(float(a[0]), float(a[1]))
        r1, c1 = grid.local_to_pixel(float(b[0]), float(b[1]))
        cv2.line(mask, (int(round(c0)), int(round(r0))), (int(round(c1)), int(round(r1))), 1, thickness)
    return mask.astype(bool)


def _segment_share_in(mask: np.ndarray, a: np.ndarray, b: np.ndarray, grid: RasterGrid) -> float:
    hits = []
    for t in np.linspace(0.0, 1.0, 11):
        point = a + (b - a) * t
        row, col = grid.local_to_pixel(float(point[0]), float(point[1]))
        row, col = int(round(row)), int(round(col))
        hits.append(0 <= row < mask.shape[0] and 0 <= col < mask.shape[1] and bool(mask[row, col]))
    return float(np.mean(hits))


def _piece(face: Any, start: np.ndarray, axis: np.ndarray, lo: float, hi: float) -> Any:
    support = np.asarray(face.points, dtype=float)
    along = (support[:, :2] - start) @ axis
    keep = (along >= lo) & (along <= hi)
    return replace(face, points=support[keep], line=LineString((tuple(start + axis * lo), tuple(start + axis * hi))))


def drop_door_leaf_faces(
    faces: list[Any], wall_points: np.ndarray, door_mask: np.ndarray, grid: RasterGrid, storey_height: float, units_per_metre: float,
    settings: DoorLeafFaceSettings = DoorLeafFaceSettings(), free_space: np.ndarray | None = None,
) -> tuple[list[Any], list[dict[str, Any]]]:
    """(faces with the free-standing leaf panels removed, removed-panel reports).

    ``wall_points`` are the measured structural points (x, y, height above the floor) the faces were fitted to; ``storey_height`` and
    those heights are plan units.  ``free_space`` (free floor raster) enables the both-sides test (always pass it in production).
    """
    if not settings.enabled or not faces or door_mask is None or not np.asarray(door_mask).any() or storey_height <= 0.0:
        return list(faces), []
    measured = np.asarray(wall_points, dtype=float)
    if measured.ndim != 2 or measured.shape[1] < 3:
        return list(faces), []
    upm = units_per_metre
    radius = max(1, int(round(settings.dilate_m * upm / grid.pixel_size)))
    doors = cv2.dilate(np.asarray(door_mask, dtype=np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1,) * 2)).astype(bool)
    free = None if free_space is None else np.asarray(free_space, dtype=bool)
    lines = [face.line if getattr(face, "kind", "") == "line" else None for face in faces]
    # pass 1: the leaf panel candidate of every face (door-class support, below the door head, leaf length, free floor both sides)
    candidates: dict[int, tuple[float, float]] = {}
    for index, face in enumerate(faces):
        support = np.asarray(getattr(face, "points", np.empty((0, 2))), dtype=float)
        if lines[index] is None or support.ndim != 2 or support.shape[1] < 2 or len(support) < 10:
            continue
        door_hit = _support_on_mask(support[:, :2], doors, grid)
        if float(door_hit.mean()) < 0.5 * settings.min_door_share:
            continue  # cheap pre-test: hardly any door-class support
        start, axis, length = _axis(face.line)
        run = _longest_run(_leaf_stretches(face, support[:, :2], door_hit, measured, settings, upm, storey_height))
        if run is None:
            continue
        step = settings.stretch_m * upm
        lo, hi = run[0] * step, min(run[1] * step, length)
        if not settings.min_leaf_m * upm <= hi - lo <= settings.max_leaf_m * upm:
            continue
        if not _free_both_sides((start + axis * lo, start + axis * hi), free, grid, settings, upm):
            continue
        candidates[index] = (lo, hi)
    # pass 2: a panel on the line (or its extension) of a collinear real wall face is a leaf / frame in a wall gap, not a stub
    walls = [line for index, line in enumerate(lines) if line is not None and index not in candidates]
    result: list[Any] = []
    dropped: list[dict[str, Any]] = []
    for index, face in enumerate(faces):
        if index not in candidates:
            result.append(face)
            continue
        start, axis, length = _axis(face.line)
        lo, hi = candidates[index]
        a, b = start + axis * lo, start + axis * hi
        if walls and _segment_share_in(_protect_mask(walls, grid, upm, settings, parallel_to=axis), a, b, grid) >= 0.5:
            result.append(face)
            continue
        for piece_lo, piece_hi in ((0.0, lo), (hi, length)):
            if piece_hi - piece_lo >= settings.min_piece_m * upm:
                result.append(_piece(face, start, axis, piece_lo, piece_hi))  # the stretch inside / beside the wall stays a face
        dropped.append({"line": [[float(a[0]), float(a[1])], [float(b[0]), float(b[1])]], "face_length_m": round(length / upm, 3), "panel_m": round((hi - lo) / upm, 3)})
    return result, dropped


def leaf_face_mask(dropped: list[dict[str, Any]], grid: RasterGrid, units_per_metre: float, half_width_m: float = 0.05, kept_faces: list[Any] | None = None, settings: DoorLeafFaceSettings = DoorLeafFaceSettings()) -> np.ndarray:
    """Raster cells of the removed leaf panels (``half_width_m`` around them), never within ``protect_m`` of a kept wall face or its extension."""
    mask = np.zeros((grid.height, grid.width), dtype=np.uint8)
    thickness = max(1, int(round(2.0 * half_width_m * units_per_metre / grid.pixel_size)))
    for item in dropped:
        (x0, y0), (x1, y1) = item["line"]
        r0, c0 = grid.local_to_pixel(x0, y0)
        r1, c1 = grid.local_to_pixel(x1, y1)
        cv2.line(mask, (int(round(c0)), int(round(r0))), (int(round(c1)), int(round(r1))), 1, thickness)
    erased = mask.astype(bool)
    kept_lines = [face.line for face in (kept_faces or []) if getattr(face, "kind", "") == "line" and getattr(face, "line", None) is not None]
    if kept_lines and erased.any():
        panels = [LineString(item["line"]) for item in dropped]
        own = [line for line in kept_lines if any(_continues(line, panel, units_per_metre) for panel in panels)]
        others = [line for line in kept_lines if not any(line is item for item in own)]
        erased &= ~_protect_mask(others, grid, units_per_metre, settings, unextended=own)
    return erased


def _continues(line: LineString, panel: LineString, upm: float) -> bool:
    """True when ``line`` is the rest of the panel's own face: collinear with it (parallel, within 3 cm) and touching its end."""
    start, axis, _ = _axis(panel)
    other_start, other_axis, _ = _axis(line)
    if abs(float(axis[0] * other_axis[1] - axis[1] * other_axis[0])) > np.sin(np.radians(3.0)):
        return False
    if abs(float((other_start - start) @ np.array((-axis[1], axis[0])))) > 0.03 * upm:
        return False
    return line.distance(panel) <= 0.02 * upm
