"""Measured indoor wall-face extraction from upper-band point evidence."""

from __future__ import annotations

from dataclasses import dataclass
import math

import cv2
import numpy as np
from shapely.geometry import LineString, Point, Polygon

from .raster_grid import RasterGrid


@dataclass(slots=True)
class WallFace:
    """A fitted straight or circular wall face and its measured support."""

    kind: str
    points: np.ndarray
    line: LineString | None = None
    center: tuple[float, float] | None = None
    radius: float | None = None
    start_angle: float | None = None
    end_angle: float | None = None
    rms: float = 0.0
    max_residual: float = 0.0
    axis_snapped: bool = False
    z_span: float = 0.0
    source: str = "measured"

    @property
    def length(self) -> float:
        if self.kind == "line" and self.line is not None:
            return float(self.line.length)
        if self.radius is not None and self.start_angle is not None and self.end_angle is not None:
            return abs(math.radians(self.end_angle - self.start_angle)) * self.radius
        return 0.0

    def geometry(self) -> LineString:
        if self.kind == "line" and self.line is not None:
            return self.line
        if self.center is None or self.radius is None or self.start_angle is None or self.end_angle is None:
            return LineString()
        angle_span = math.radians(self.end_angle - self.start_angle)
        count = max(8, int(math.ceil(abs(angle_span) * self.radius / 0.02)))
        angles = np.linspace(math.radians(self.start_angle), math.radians(self.end_angle), count)
        return LineString([(self.center[0] + self.radius * math.cos(angle), self.center[1] + self.radius * math.sin(angle)) for angle in angles])


def extract_wall_faces(
    evidence_points: np.ndarray,
    interior_mask: np.ndarray,
    grid: RasterGrid,
    *,
    units_per_metre: float,
    min_length: float = 0.30,
) -> list[WallFace]:
    """Fit wall faces from vertically persistent structural raster cells."""
    raw = np.asarray(evidence_points, dtype=float)
    points = raw[:, :2].reshape((-1, 2)) if raw.size else np.empty((0, 2), dtype=float)
    if not len(points):
        return []
    mask = np.asarray(interior_mask, dtype=bool)
    if mask.shape != (grid.height, grid.width):
        raise ValueError("interior_mask shape must match RasterGrid")
    # Dense, noisily scanned orthogonal room boundaries are better described by
    # the projection extractor than by sequential RANSAC.  Keep the curved
    # RANSAC path for all other inputs.
    if raw.ndim == 2 and raw.shape[1] == 2:
        projected = extract_axis_wall_faces(raw, grid, units_per_metre=units_per_metre)
        if len(projected) >= 6:
            return [face for face in projected if face.length >= min_length * units_per_metre]
    candidates = points
    weights = raw[:, 2] if raw.ndim == 2 and raw.shape[1] >= 3 else np.ones(len(points), dtype=float)
    spans = raw[:, 3] if raw.ndim == 2 and raw.shape[1] >= 4 else np.full(len(points), 1.0 * units_per_metre)
    if len(candidates) < 2:
        return []
    # Preserve the 5 mm evidence contract at ingestion, while fitting from one
    # representative per centimetre cell keeps RANSAC bounded on dense scans.
    cells = np.floor(candidates / (0.01 * units_per_metre)).astype(np.int64)
    keys = (cells[:, 0] << 32) ^ (cells[:, 1] & np.int64(0xFFFFFFFF))
    _keys, representative = np.unique(keys, return_index=True)
    selected = np.sort(representative)
    candidates, weights, spans = candidates[selected], weights[selected], spans[selected]
    if raw.ndim == 2 and raw.shape[1] == 2:
        faces = [face for cluster in _spatial_clusters(candidates, max(0.06 * units_per_metre, grid.pixel_size * 3.0)) for face in _extract_cluster_faces(cluster, units_per_metre, min_length, weights=weights[:len(cluster)], spans=spans[:len(cluster)])]
    else:
        faces = _extract_cluster_faces(candidates, units_per_metre, min_length, weights=weights, spans=spans)
    faces = _snap_and_close(faces, units_per_metre, strict=raw.ndim == 2 and raw.shape[1] >= 4)
    return [face for face in faces if face.length >= min_length * units_per_metre]


def extract_axis_wall_faces(evidence: np.ndarray, grid: RasterGrid, *, units_per_metre: float, passable: np.ndarray | None = None) -> list[WallFace]:
    """Recover measured orthogonal faces with the validated projection method.

    ``evidence`` contains every structural point as ``x, y, z_above_floor``.
    Keeping the individual heights is essential: the occupancy test rejects
    horizontal clutter without discarding the two physical faces of a wall.

    ``passable`` is an optional free-floor mask on ``grid``.  Gaps of 0.6-1.0 m
    are bridged only when it shows that the gap is not walkable from both
    sides; without it such door-sized gaps always stay open.
    """
    data = np.asarray(evidence, dtype=float)
    if data.ndim != 2 or data.shape[0] < 2:
        return []
    if data.shape[1] < 3:
        # Raster-only fallback: retain the public helper's useful behaviour.
        data = np.repeat(data[:, :2], 3, axis=0)
        data = np.column_stack((data, np.tile(np.asarray((0.30, 0.60, 0.90)) * units_per_metre, len(evidence))))
    points, heights = data[:, :2], data[:, 2]
    theta = _projection_axis_angle(points)
    rotation = np.asarray(((math.cos(theta), math.sin(theta)), (-math.sin(theta), math.cos(theta))))
    rotated = points @ rotation.T
    along_bin, across_bin = 0.06 * units_per_metre, 0.01 * units_per_metre
    band, minimum_points = 2, 3
    gap, minimum_length, minimum_fraction = 0.25 * units_per_metre, 0.40 * units_per_metre, 0.60
    side_start, side_end, face_fraction = 0.07 * units_per_metre, 0.18 * units_per_metre, 0.50
    nms = 0.15 * units_per_metre
    faces: list[WallFace] = []
    for axis_index in (0, 1):
        along, across = rotated[:, axis_index], rotated[:, 1 - axis_index]
        along0, across0 = float(along.min()), float(across.min())
        ia = np.floor((along - along0) / along_bin).astype(np.int32)
        ic = np.floor((across - across0) / across_bin).astype(np.int32)
        count = np.zeros((int(ic.max()) + 1, int(ia.max()) + 1), dtype=np.int32)
        np.add.at(count, (ic, ia), 1)
        z_index = np.clip(np.floor(heights / (0.10 * units_per_metre)).astype(np.int32), 0, 40)
        zmask = np.zeros_like(count, dtype=np.int64)
        np.bitwise_or.at(zmask, (ic, ia), (np.int64(1) << z_index).astype(np.int64))
        band_count = cv2.filter2D(count.astype(np.float32), -1, np.ones((2 * band + 1, 1), dtype=np.float32), borderType=cv2.BORDER_CONSTANT)
        band_zmask = zmask.copy()
        for delta in range(1, band + 1):
            band_zmask[delta:] |= zmask[:-delta]
            band_zmask[:-delta] |= zmask[delta:]
        vertical_bins = _bit_population(band_zmask)
        occupied = (band_count >= minimum_points) & (vertical_bins >= 3)
        candidates: list[tuple[float, int, int, int]] = []
        for row in range(occupied.shape[0]):
            row_occupied = occupied[row]
            indices = np.flatnonzero(row_occupied)
            if not len(indices):
                continue
            start = previous = int(indices[0])
            for index in [*map(int, indices[1:]), None]:
                if index is not None and (index - previous) * along_bin <= gap:
                    previous = index
                    continue
                length = (previous - start + 1) * along_bin
                fraction = float(row_occupied[start:previous + 1].mean())
                if length >= minimum_length and fraction >= minimum_fraction:
                    lower = count[max(0, row - int(math.ceil(side_end / across_bin))):max(0, row - int(math.floor(side_start / across_bin)) + 1), start:previous + 1].sum(axis=0)
                    upper = count[row + int(math.floor(side_start / across_bin)):row + int(math.ceil(side_end / across_bin)) + 1, start:previous + 1].sum(axis=0)
                    if max(float((lower == 0).mean()), float((upper == 0).mean())) >= face_fraction:
                        candidates.append((float(band_count[row, start:previous + 1].sum()), row, start, previous))
                if index is not None:
                    start = previous = index
        candidates.sort(reverse=True)
        selected: list[tuple[int, int, int]] = []
        for support, row, start, end in candidates:
            if any(abs(row - kept_row) * across_bin <= nms and min(end, kept_end) - max(start, kept_start) > 0 for kept_row, kept_start, kept_end in selected):
                continue
            selected.append((row, start, end))
            support_mask = (np.abs(ic - row) <= band) & (ia >= start) & (ia <= end)
            fitted_across = float(np.median(across[support_mask])) if support_mask.any() else across0 + row * across_bin
            residuals = np.abs(across[support_mask] - fitted_across)
            first, last = along0 + start * along_bin, along0 + (end + 1) * along_bin
            rotated_endpoints = [(first, fitted_across), (last, fitted_across)] if axis_index == 0 else [(fitted_across, first), (fitted_across, last)]
            line = LineString(np.asarray(rotated_endpoints) @ rotation)
            faces.append(WallFace("line", points[support_mask], line=line, rms=_rms(residuals), max_residual=float(np.percentile(residuals, 90.0)) if len(residuals) else 0.0, axis_snapped=True, z_span=float(np.ptp(heights[support_mask])) if support_mask.any() else 0.0))
    _merge_axis_faces(faces, units_per_metre, grid=grid, passable=passable)
    _close_corners(faces, 0.30 * units_per_metre)
    _drop_isolated_short_faces(faces, units_per_metre)
    return [face for face in faces if face.line is not None and face.length >= 0.40 * units_per_metre]


def _projection_axis_angle(points: np.ndarray) -> float:
    """Find the dominant orthogonal axes by exhaustive projection scoring."""
    def score(degrees: float) -> float:
        angle = math.radians(degrees)
        u = points @ np.asarray((math.cos(angle), math.sin(angle)))
        v = points @ np.asarray((-math.sin(angle), math.cos(angle)))
        return float(np.square(np.bincount(np.floor((u - u.min()) / 0.02).astype(np.int32))).sum() + np.square(np.bincount(np.floor((v - v.min()) / 0.02).astype(np.int32))).sum())
    coarse = np.arange(0.0, 90.0, 0.25)
    best = float(coarse[int(np.argmax([score(value) for value in coarse]))])
    refined = np.arange(best - 0.25, best + 0.2501, 0.05)
    return math.radians(float(refined[int(np.argmax([score(value) for value in refined]))]))


def _bit_population(values: np.ndarray) -> np.ndarray:
    remaining = values.copy()
    count = np.zeros_like(values, dtype=np.int16)
    while remaining.any():
        count += (remaining & 1).astype(np.int16)
        remaining >>= 1
    return count


def _merge_axis_faces(faces: list[WallFace], units_per_metre: float, *, grid: RasterGrid | None = None, passable: np.ndarray | None = None) -> None:
    """Merge scan gaps, while preserving door-sized gaps for the opening pass."""
    changed = True
    while changed:
        changed = False
        for index, first in enumerate(faces):
            if first.line is None:
                continue
            first_axis = np.asarray(first.line.coords[-1]) - np.asarray(first.line.coords[0])
            first_axis /= max(float(np.linalg.norm(first_axis)), 1e-12)
            normal = np.asarray((-first_axis[1], first_axis[0]))
            for second in faces[index + 1:]:
                if second.line is None:
                    continue
                second_axis = np.asarray(second.line.coords[-1]) - np.asarray(second.line.coords[0])
                second_axis /= max(float(np.linalg.norm(second_axis)), 1e-12)
                if abs(float(_cross2(first_axis, second_axis))) > math.sin(math.radians(1.0)):
                    continue
                offset = abs(float(np.dot(np.asarray(second.line.coords[0]) - np.asarray(first.line.coords[0]), normal)))
                if offset > 0.02 * units_per_metre:
                    continue
                endpoints = [np.asarray(point) for point in first.line.coords] + [np.asarray(point) for point in second.line.coords]
                gap = min(float(np.linalg.norm(np.asarray(left) - np.asarray(right))) for left in first.line.coords for right in second.line.coords)
                # Scan gaps below a door width are always bridged.  Gaps of
                # 0.6-1.0 m are bridged only where the free floor shows they are
                # not a doorway (walkable on both sides and inside the gap).
                if gap > 1.0 * units_per_metre:
                    continue
                if gap >= 0.60 * units_per_metre and _gap_is_doorway(first, second, normal, units_per_metre, grid, passable):
                    continue
                anchor = np.asarray(first.line.coords[0])
                projections = [float(np.dot(point - anchor, first_axis)) for point in endpoints]
                first.line = LineString([anchor + first_axis * min(projections), anchor + first_axis * max(projections)])
                first.points = np.vstack((first.points, second.points))
                second.line = None
                changed = True
                break
            if changed:
                break
    faces[:] = [face for face in faces if face.line is not None]


def _gap_is_doorway(first: WallFace, second: WallFace, normal: np.ndarray, units_per_metre: float, grid: RasterGrid | None, passable: np.ndarray | None) -> bool:
    """True when a collinear gap may be a doorway (always, without a floor mask)."""
    if grid is None or passable is None:
        return True
    mask = np.asarray(passable, dtype=bool)
    if mask.shape != (grid.height, grid.width):
        return True
    pairs = [(np.asarray(left), np.asarray(right)) for left in first.line.coords for right in second.line.coords]
    left, right = min(pairs, key=lambda pair: float(np.linalg.norm(pair[0] - pair[1])))
    centre = (left + right) / 2.0
    radius = max(1, int(round(0.05 * units_per_metre / grid.pixel_size)))

    def walkable(point: np.ndarray) -> bool:
        row, col = grid.local_to_pixel(float(point[0]), float(point[1]))
        row, col = int(round(row)), int(round(col))
        window = mask[max(0, row - radius):max(0, row + radius + 1), max(0, col - radius):max(0, col + radius + 1)]
        return bool(window.size) and bool(window.any())

    offset = 0.30 * units_per_metre
    return walkable(centre) and walkable(centre + normal * offset) and walkable(centre - normal * offset)


def line_is_doorway(line: LineString, normal: np.ndarray, units_per_metre: float, grid: RasterGrid | None, passable: np.ndarray | None) -> bool:
    """Return whether a proposed ceiling wall piece is an open doorway gap."""
    if grid is None or passable is None or line.is_empty:
        return False
    mask = np.asarray(passable, dtype=bool)
    if mask.shape != (grid.height, grid.width):
        return False
    offset = 0.30 * units_per_metre
    samples = [line.interpolate(value, normalized=True) for value in np.linspace(0.15, 0.85, 5)]
    def walkable(point: Point) -> bool:
        row, col = grid.local_to_pixel(float(point.x), float(point.y))
        radius = max(1, int(round(0.05 * units_per_metre / grid.pixel_size)))
        r0, r1 = max(0, int(round(row)) - radius), min(mask.shape[0], int(round(row)) + radius + 1)
        c0, c1 = max(0, int(round(col)) - radius), min(mask.shape[1], int(round(col)) + radius + 1)
        return bool(mask[r0:r1, c0:c1].any())
    open_samples = [walkable(point) and walkable(Point(point.x + normal[0] * offset, point.y + normal[1] * offset)) and walkable(Point(point.x - normal[0] * offset, point.y - normal[1] * offset)) for point in samples]
    return bool(open_samples) and float(np.mean(open_samples)) >= 0.50


def _drop_isolated_short_faces(faces: list[WallFace], units_per_metre: float) -> None:
    kept: list[WallFace] = []
    for face in faces:
        if face.line is None or face.length >= 0.60 * units_per_metre:
            kept.append(face)
            continue
        if any(other is not face and other.line is not None and face.line.distance(other.line) <= 0.30 * units_per_metre for other in faces):
            kept.append(face)
    faces[:] = kept


def face_buffer_polygons(faces: list[WallFace], width: float) -> list[Polygon]:
    """Create thin wall proxies for the existing opening detector."""
    buffers: list[Polygon] = []
    for face in faces:
        geometry = face.geometry()
        if geometry.is_empty or geometry.length <= 1e-9:
            continue
        buffered = geometry.buffer(width, cap_style=2, join_style=2)
        if isinstance(buffered, Polygon) and not buffered.is_empty and buffered.area > 1e-12:
            buffers.append(buffered)
    return buffers


def break_faces_at_openings(faces: list[WallFace], openings: list[object], tolerance: float) -> list[WallFace]:
    """Split straight faces at measured door and window spans."""
    output: list[WallFace] = []
    for face in faces:
        if face.kind != "line" or face.line is None:
            output.append(face)
            continue
        cuts: list[tuple[float, float]] = []
        for opening in openings:
            point = Point(getattr(opening, "center"))
            if face.line.distance(point) > tolerance:
                continue
            centre = face.line.project(point)
            width = float(getattr(opening, "width"))
            cuts.append((max(0.0, centre - width / 2.0), min(face.line.length, centre + width / 2.0)))
        if not cuts:
            output.append(face)
            continue
        cursor = 0.0
        for start, end in sorted(cuts):
            if start - cursor >= 1e-6:
                output.append(_face_segment(face, cursor, start))
            cursor = max(cursor, end)
        if face.line.length - cursor >= 1e-6:
            output.append(_face_segment(face, cursor, face.line.length))
    return [face for face in output if face.length >= 0.02]


def extend_faces_for_window_gaps(faces: list[WallFace], windows: list[object], evidence: np.ndarray, grid: RasterGrid, units_per_metre: float, *, passable: np.ndarray | None = None) -> tuple[list[WallFace], int, float]:
    """Close short, supported wall gaps on a window's measured wall alignment.

    The window spans themselves are never filled.  Only the intervening runs to
    an existing wall end or the next window jamb become faces.
    """
    source = list(faces)
    additions: list[WallFace] = []
    pending = [window for window in windows if getattr(window, "kind", "") == "WINDOW"]
    consumed: set[int] = set()
    for index, seed in enumerate(pending):
        if index in consumed:
            continue
        axis = np.asarray((math.cos(float(seed.axis_angle)), math.sin(float(seed.axis_angle))))
        centre = np.asarray(seed.center, dtype=float)
        aligned = [face for face in source if face.kind == "line" and face.line is not None and _parallel_axes(axis, _line_direction(face.line), 5.0) and _infinite_line_offset(face.line, centre) <= 0.02 * units_per_metre]
        if not aligned:
            continue
        group = [seed]
        consumed.add(index)
        for other_index, other in enumerate(pending[index + 1:], start=index + 1):
            if other_index in consumed:
                continue
            other_axis = np.asarray((math.cos(float(other.axis_angle)), math.sin(float(other.axis_angle))))
            if _parallel_axes(axis, other_axis, 5.0) and abs(float(np.cross(np.asarray(other.center) - centre, axis))) <= 0.02 * units_per_metre:
                group.append(other)
                consumed.add(other_index)
        additions.extend(_window_gap_additions(aligned, group, axis, evidence, grid, units_per_metre, passable, source))
    unique: list[WallFace] = []
    for face in additions:
        if face.line is not None and not any(face.line.distance(other.line) <= 0.01 * units_per_metre and _parallel_axes(_line_direction(face.line), _line_direction(other.line), 2.0) and _line_overlap_extent(face.line, other.line) >= 0.90 * min(face.line.length, other.line.length) for other in [*source, *unique] if other.line is not None):
            unique.append(face)
    return [*source, *unique], len(unique), float(sum(face.length for face in unique))


def _window_gap_additions(aligned: list[WallFace], windows: list[object], axis: np.ndarray, evidence: np.ndarray, grid: RasterGrid, units_per_metre: float, passable: np.ndarray | None, source: list[WallFace]) -> list[WallFace]:
    """Fill the non-opening intervals between face ends and window jambs."""
    reference = max(aligned, key=lambda face: face.line.length)
    origin = np.asarray(reference.line.coords[0], dtype=float)
    normal = np.asarray((-axis[1], axis[0]))
    face_intervals = [(min(float(np.dot(np.asarray(point) - origin, axis)) for point in face.line.coords), max(float(np.dot(np.asarray(point) - origin, axis)) for point in face.line.coords)) for face in aligned]
    window_intervals = [(float(np.dot(np.asarray(window.center) - origin, axis)) - float(window.width) / 2.0, float(np.dot(np.asarray(window.center) - origin, axis)) + float(window.width) / 2.0) for window in windows]
    boundaries = sorted({value for interval in [*face_intervals, *window_intervals] for value in interval})
    additions: list[WallFace] = []
    for start, end in zip(boundaries, boundaries[1:], strict=False):
        midpoint = (start + end) / 2.0
        if any(first - 1e-9 <= midpoint <= second + 1e-9 for first, second in face_intervals + window_intervals):
            continue
        candidate = LineString((tuple(origin + axis * start), tuple(origin + axis * end)))
        if candidate.length <= 0.02 * units_per_metre or candidate.length > 2.0 * units_per_metre:
            continue
        if line_is_doorway(candidate, normal, units_per_metre, grid, passable):
            continue
        endpoints = (np.asarray(candidate.coords[0]), np.asarray(candidate.coords[-1]))
        joins_window = any(min(abs(value - edge) for edge in interval) <= 1e-6 for value in (start, end) for interval in window_intervals)
        joins_corner = any(_endpoint_is_corner(point, source, axis, units_per_metre) for point in endpoints)
        if _line_evidence_fraction(candidate, evidence, grid, 0.02 * units_per_metre) < 0.30 and not (joins_window and joins_corner):
            continue
        additions.append(WallFace("line", np.empty((0, 2), dtype=float), line=candidate, axis_snapped=True, source="window_extension"))
    return additions


def _line_direction(line: LineString) -> np.ndarray:
    vector = np.asarray(line.coords[-1], dtype=float) - np.asarray(line.coords[0], dtype=float)
    return vector / max(float(np.linalg.norm(vector)), 1e-12)


def _infinite_line_offset(line: LineString, point: np.ndarray) -> float:
    direction = _line_direction(line)
    normal = np.asarray((-direction[1], direction[0]))
    return abs(float(np.dot(np.asarray(point, dtype=float) - np.asarray(line.coords[0], dtype=float), normal)))


def _parallel_axes(first: np.ndarray, second: np.ndarray, tolerance_deg: float) -> bool:
    return abs(math.degrees(math.asin(float(np.clip(abs(np.cross(first, second)), 0.0, 1.0))))) <= tolerance_deg


def _line_overlap_extent(first: LineString, second: LineString) -> float:
    direction = _line_direction(first)
    origin = np.asarray(first.coords[0], dtype=float)
    values_first = [float(np.dot(np.asarray(point) - origin, direction)) for point in first.coords]
    values_second = [float(np.dot(np.asarray(point) - origin, direction)) for point in second.coords]
    return max(0.0, min(max(values_first), max(values_second)) - max(min(values_first), min(values_second)))


def _line_evidence_fraction(line: LineString, evidence: np.ndarray, grid: RasterGrid, radius: float) -> float:
    samples = [line.interpolate(value, normalized=True) for value in np.linspace(0.0, 1.0, max(3, int(math.ceil(line.length / max(grid.pixel_size, 1e-9))) + 1))]
    values = []
    pixels = max(1, int(math.ceil(radius / grid.pixel_size)))
    for point in samples:
        row, col = grid.local_to_pixel(float(point.x), float(point.y))
        row, col = int(round(row)), int(round(col))
        values.append(bool(evidence[max(0, row - pixels):min(grid.height, row + pixels + 1), max(0, col - pixels):min(grid.width, col + pixels + 1)].any()))
    return float(np.mean(values)) if values else 0.0


def _endpoint_is_corner(point: np.ndarray, faces: list[WallFace], axis: np.ndarray, units_per_metre: float) -> bool:
    for face in faces:
        if face.kind != "line" or face.line is None or face.line.distance(Point(point)) > 0.05 * units_per_metre:
            continue
        angle = abs(math.degrees(math.asin(float(np.clip(abs(np.cross(axis, _line_direction(face.line))), 0.0, 1.0)))))
        if 80.0 <= angle <= 100.0:
            return True
    return False


def face_entities(faces: list[WallFace]) -> list[dict[str, object]]:
    """Convert measured faces to the neutral DXF entity format."""
    entities: list[dict[str, object]] = []
    for face in faces:
        if face.kind == "line" and face.line is not None:
            entities.append({"type": "polyline", "points": [[float(x), float(y)] for x, y in face.line.coords], "closed": False, "_rms": face.rms, "_max_residual": face.max_residual, "_axis_snapped": face.axis_snapped, "_source": face.source})
        elif face.kind == "arc" and face.center is not None and face.radius is not None and face.start_angle is not None and face.end_angle is not None:
            entities.append({"type": "arc", "center": [float(face.center[0]), float(face.center[1])], "radius": float(face.radius), "start_angle": float(face.start_angle), "end_angle": float(face.end_angle), "_rms": face.rms, "_max_residual": face.max_residual, "_axis_snapped": False})
    return entities


def _spatial_clusters(points: np.ndarray, radius: float) -> list[np.ndarray]:
    cells = np.floor(points / radius).astype(int)
    lookup: dict[tuple[int, int], list[int]] = {}
    for index, cell in enumerate(cells):
        lookup.setdefault((int(cell[0]), int(cell[1])), []).append(index)
    visited: set[tuple[int, int]] = set()
    clusters: list[np.ndarray] = []
    for cell in lookup:
        if cell in visited:
            continue
        pending = [cell]
        indices: list[int] = []
        visited.add(cell)
        while pending:
            current = pending.pop()
            indices.extend(lookup[current])
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    neighbour = (current[0] + dx, current[1] + dy)
                    if neighbour in lookup and neighbour not in visited:
                        visited.add(neighbour)
                        pending.append(neighbour)
        clusters.append(points[np.asarray(indices, dtype=int)])
    return clusters


def _extract_cluster_faces(points: np.ndarray, units_per_metre: float, min_length: float, *, weights: np.ndarray | None = None, spans: np.ndarray | None = None) -> list[WallFace]:
    remaining = np.asarray(points, dtype=float)
    remaining_weights = np.ones(len(remaining), dtype=float) if weights is None else np.asarray(weights, dtype=float)
    remaining_spans = np.full(len(remaining), units_per_metre, dtype=float) if spans is None else np.asarray(spans, dtype=float)
    result: list[WallFace] = []
    # Prefer a coherent curved run before sequential line extraction can break
    # a smooth arc into a series of short tangent chords.
    circle, circle_indices = _ransac_circle(remaining, 0.015 * units_per_metre) if len(remaining) >= 12 else (None, np.empty(0, dtype=int))
    if circle is not None and len(circle_indices) >= max(12, int(len(remaining) * 0.30)):
        support = remaining[circle_indices]
        center, radius = _fit_circle(support)
        residuals = np.abs(np.hypot(support[:, 0] - center[0], support[:, 1] - center[1]) - radius)
        angles = _arc_angles(support, center)
        span = float(np.max(angles) - np.min(angles))
        sagitta = radius * (1.0 - math.cos(min(abs(span), math.pi * 2.0) / 2.0))
        line_max = float(np.max(_line_residuals(support, _fit_line(support))))
        continuous = _max_support_gap(support, _fit_line(support)) <= 0.15 * units_per_metre
        if support.shape[0] >= 20 and radius <= 15.0 * units_per_metre and sagitta >= 0.05 * units_per_metre and abs(float(np.max(angles) - np.min(angles)) * radius) >= 1.0 * units_per_metre and continuous and _rms(residuals) <= 0.01 * units_per_metre and _rms(residuals) * 2.0 <= _rms(_line_residuals(support, _fit_line(support))):
            result.append(WallFace("arc", support, center=(float(center[0]), float(center[1])), radius=float(radius), start_angle=math.degrees(float(np.min(angles))), end_angle=math.degrees(float(np.max(angles))), rms=_rms(residuals), max_residual=float(np.max(residuals)), z_span=float(np.median(remaining_spans[circle_indices]))))
            remaining = np.delete(remaining, circle_indices, axis=0)
            remaining_weights = np.delete(remaining_weights, circle_indices)
            remaining_spans = np.delete(remaining_spans, circle_indices)
    while len(remaining) >= 8:
        line, indices = _ransac_line(remaining, 0.02 * units_per_metre, remaining_weights)
        if line is None or len(indices) < 8:
            break
        support = remaining[indices]
        fitted = _fit_line(support)
        support = support[_line_residuals(support, fitted) <= 0.015 * units_per_metre]
        if len(support) < 8:
            remaining, remaining_weights, remaining_spans = np.delete(remaining, indices, axis=0), np.delete(remaining_weights, indices), np.delete(remaining_spans, indices)
            continue
        fitted = _fit_line(support)
        residuals = _line_residuals(support, fitted)
        for fragment in _continuous_line_fragments(support, fitted, 0.30 * units_per_metre):
            fragment_fit = _fit_line(fragment)
            fragment_residuals = _line_residuals(fragment, fragment_fit)
            coverage = _line_coverage(fragment, fragment_fit, 0.01 * units_per_metre)
            if fragment_fit.length >= max(0.40, min_length) * units_per_metre and coverage >= 0.60 and _rms(fragment_residuals) <= 0.01 * units_per_metre and float(np.max(fragment_residuals)) <= 0.02 * units_per_metre:
                result.append(WallFace("line", fragment, line=fragment_fit, rms=_rms(fragment_residuals), max_residual=float(np.max(fragment_residuals)), z_span=float(np.median(remaining_spans[indices]))))
        remaining, remaining_weights, remaining_spans = np.delete(remaining, indices, axis=0), np.delete(remaining_weights, indices), np.delete(remaining_spans, indices)
    # A true curve remains after the straight runs have been consumed.
    while len(remaining) >= 12:
        circle, indices = _ransac_circle(remaining, 0.015 * units_per_metre)
        if circle is None or len(indices) < 12:
            break
        support = remaining[indices]
        center, radius = _fit_circle(support)
        residuals = np.abs(np.hypot(support[:, 0] - center[0], support[:, 1] - center[1]) - radius)
        angles = _arc_angles(support, center)
        span = float(np.max(angles) - np.min(angles))
        sagitta = radius * (1.0 - math.cos(min(abs(span), math.pi * 2.0) / 2.0))
        line_fit = _fit_line(support)
        line_max = float(np.max(_line_residuals(support, line_fit)))
        continuous = _max_support_gap(support, line_fit) <= 0.15 * units_per_metre
        if support.shape[0] >= 20 and radius <= 15.0 * units_per_metre and sagitta >= 0.05 * units_per_metre and abs(span * radius) >= 1.0 * units_per_metre and continuous and _rms(residuals) <= 0.01 * units_per_metre and _rms(residuals) * 2.0 <= _rms(_line_residuals(support, line_fit)):
            start, end = math.degrees(float(np.min(angles))), math.degrees(float(np.max(angles)))
            result.append(WallFace("arc", support, center=(float(center[0]), float(center[1])), radius=float(radius), start_angle=start, end_angle=end, rms=_rms(residuals), max_residual=float(np.max(residuals))))
        remaining, remaining_weights, remaining_spans = np.delete(remaining, indices, axis=0), np.delete(remaining_weights, indices), np.delete(remaining_spans, indices)
    return result


def _ransac_line(points: np.ndarray, threshold: float, weights: np.ndarray | None = None) -> tuple[LineString | None, np.ndarray]:
    rng = np.random.default_rng(20261111)
    best = np.empty(0, dtype=int)
    best_score = -1.0
    for _ in range(min(512, max(32, len(points) * 3))):
        pair = rng.choice(len(points), 2, replace=False)
        line = LineString(points[pair])
        if line.length <= 1e-9:
            continue
        direction = np.asarray(line.coords[1]) - np.asarray(line.coords[0])
        residuals = np.abs(_cross2(direction, points - np.asarray(line.coords[0]))) / np.linalg.norm(direction)
        inliers = np.flatnonzero(residuals <= threshold)
        score = float(np.sum(weights[inliers])) if weights is not None else float(len(inliers))
        if score > best_score:
            best, best_score = inliers, score
    return (_fit_line(points[best]) if len(best) >= 2 else None), best


def _ransac_circle(points: np.ndarray, threshold: float) -> tuple[tuple[np.ndarray, float] | None, np.ndarray]:
    rng = np.random.default_rng(20261112)
    best = np.empty(0, dtype=int)
    candidate: tuple[np.ndarray, float] | None = None
    for _ in range(min(512, max(64, len(points) * 4))):
        sample = points[rng.choice(len(points), 3, replace=False)]
        try:
            center, radius = _fit_circle(sample)
        except np.linalg.LinAlgError:
            continue
        residuals = np.abs(np.hypot(points[:, 0] - center[0], points[:, 1] - center[1]) - radius)
        inliers = np.flatnonzero(residuals <= threshold)
        if len(inliers) > len(best):
            best, candidate = inliers, (center, radius)
    return candidate, best


def _fit_line(points: np.ndarray) -> LineString:
    centre = np.mean(points, axis=0)
    _values, vectors = np.linalg.eigh(np.cov((points - centre).T))
    direction = vectors[:, -1]
    projections = (points - centre) @ direction
    return LineString([centre + direction * float(np.min(projections)), centre + direction * float(np.max(projections))])


def _fit_circle(points: np.ndarray) -> tuple[np.ndarray, float]:
    matrix = np.column_stack((2.0 * points[:, 0], 2.0 * points[:, 1], np.ones(len(points))))
    rhs = np.sum(points * points, axis=1)
    solution, *_rest = np.linalg.lstsq(matrix, rhs, rcond=None)
    center = solution[:2]
    radius = math.sqrt(max(0.0, float(solution[2] + center @ center)))
    return center, radius


def _line_residuals(points: np.ndarray, line: LineString) -> np.ndarray:
    start, end = np.asarray(line.coords[0]), np.asarray(line.coords[-1])
    direction = end - start
    return np.abs(_cross2(direction, points - start)) / max(float(np.linalg.norm(direction)), 1e-12)


def _continuous_line_fragments(points: np.ndarray, line: LineString, maximum_gap: float) -> list[np.ndarray]:
    axis = np.asarray(line.coords[-1]) - np.asarray(line.coords[0])
    axis /= max(float(np.linalg.norm(axis)), 1e-12)
    order = np.argsort((points - np.asarray(line.coords[0])) @ axis)
    ordered = points[order]
    projections = (ordered - np.asarray(line.coords[0])) @ axis
    cuts = np.flatnonzero(np.diff(projections) > maximum_gap) + 1
    return [fragment for fragment in np.split(ordered, cuts) if len(fragment) >= 2]


def _max_support_gap(points: np.ndarray, line: LineString) -> float:
    fragments = _continuous_line_fragments(points, line, float("inf"))
    if not fragments:
        return float("inf")
    axis = np.asarray(line.coords[-1]) - np.asarray(line.coords[0])
    axis /= max(float(np.linalg.norm(axis)), 1e-12)
    values = np.sort((points - np.asarray(line.coords[0])) @ axis)
    return float(np.max(np.diff(values))) if len(values) >= 2 else float("inf")


def _line_coverage(points: np.ndarray, line: LineString, resolution: float) -> float:
    if line.length <= 1e-9:
        return 0.0
    axis = np.asarray(line.coords[-1]) - np.asarray(line.coords[0])
    axis /= line.length
    values = (points - np.asarray(line.coords[0])) @ axis
    bins = np.unique(np.floor(values / resolution).astype(int))
    return float(len(bins) * resolution / line.length)


def _rms(values: np.ndarray) -> float:
    return float(math.sqrt(float(np.mean(values * values)))) if len(values) else 0.0


def _arc_angles(points: np.ndarray, center: np.ndarray) -> np.ndarray:
    """Order support angles around the smallest circular interval they occupy."""
    raw = np.sort(np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0]))
    if len(raw) <= 1:
        return raw
    gaps = np.diff(np.concatenate((raw, [raw[0] + math.tau])))
    cut = int(np.argmax(gaps))
    return np.concatenate((raw[cut + 1:], raw[:cut + 1] + math.tau))


def _snap_and_close(faces: list[WallFace], units_per_metre: float, *, strict: bool = True) -> list[WallFace]:
    lines = [face for face in faces if face.kind == "line" and face.line is not None]
    if not lines:
        return faces
    angles = np.asarray([math.atan2(face.line.coords[-1][1] - face.line.coords[0][1], face.line.coords[-1][0] - face.line.coords[0][0]) % math.pi for face in lines])
    weights = np.asarray([face.length for face in lines])
    bins = np.linspace(0.0, math.pi, 73)
    histogram, _edges = np.histogram(angles, bins=bins, weights=weights)
    primary = float((bins[int(np.argmax(histogram))] + bins[int(np.argmax(histogram)) + 1]) / 2.0)
    axes = (np.asarray((math.cos(primary), math.sin(primary))), np.asarray((-math.sin(primary), math.cos(primary))))
    for face in lines:
        direction = np.asarray(face.line.coords[-1]) - np.asarray(face.line.coords[0])
        direction /= max(float(np.linalg.norm(direction)), 1e-12)
        axis = min(axes, key=lambda candidate: abs(math.asin(float(np.clip(abs(_cross2(direction, candidate)), 0.0, 1.0)))))
        angle = abs(math.asin(float(np.clip(abs(_cross2(direction, axis)), 0.0, 1.0))))
        centre = np.mean(face.points, axis=0)
        projections = (face.points - centre) @ axis
        snapped = LineString([centre + axis * float(np.min(projections)), centre + axis * float(np.max(projections))])
        residuals = _line_residuals(face.points, snapped)
        if angle <= math.radians(3.0) and float(np.percentile(residuals, 90.0)) <= 0.02 * units_per_metre:
            face.line, face.rms, face.max_residual, face.axis_snapped = snapped, _rms(residuals), float(np.max(residuals)), True
        elif strict and (face.length < 1.0 * units_per_metre or face.z_span < 0.5 * units_per_metre):
            face.line = None
    _merge_collinear(lines, 0.02 * units_per_metre, 0.30 * units_per_metre)
    _close_corners(lines, 0.30 * units_per_metre)
    return faces


def _merge_collinear(lines: list[WallFace], offset: float, gap: float) -> None:
    changed = True
    while changed:
        changed = False
        for index, first in enumerate(lines):
            if first.line is None:
                continue
            axis = np.asarray(first.line.coords[-1]) - np.asarray(first.line.coords[0])
            axis /= max(float(np.linalg.norm(axis)), 1e-12)
            normal = np.asarray((-axis[1], axis[0]))
            for second in lines[index + 1:]:
                if second.line is None:
                    continue
                other = np.asarray(second.line.coords[-1]) - np.asarray(second.line.coords[0])
                other /= max(float(np.linalg.norm(other)), 1e-12)
                if abs(math.asin(float(np.clip(abs(_cross2(axis, other)), 0.0, 1.0)))) > math.radians(1.0):
                    continue
                if abs(float(np.dot(np.asarray(second.line.centroid.coords[0]) - np.asarray(first.line.coords[0]), normal))) > offset:
                    continue
                values = [float(np.dot(np.asarray(point), axis)) for point in list(first.line.coords) + list(second.line.coords)]
                if min(math.dist(a, b) for a in first.line.coords for b in second.line.coords) > gap and max(values) - min(values) > first.length + second.length + gap:
                    continue
                anchor = np.asarray(first.line.coords[0])
                first.line = LineString([anchor + axis * (min(values) - np.dot(anchor, axis)), anchor + axis * (max(values) - np.dot(anchor, axis))])
                first.points = np.vstack((first.points, second.points))
                second.line = None
                changed = True
                break
            if changed:
                break
    lines[:] = [line for line in lines if line.line is not None]


def _close_corners(lines: list[WallFace], distance: float) -> None:
    # Corners are only closed between clearly non-parallel faces.  Snapping to
    # a parallel (offset) neighbour would bend a face into a diagonal stub.
    minimum_sine = math.sin(math.radians(60.0))
    for index, first in enumerate(lines):
        if first.line is None:
            continue
        for second in lines[index + 1:]:
            if second.line is None:
                continue
            a0, a1 = np.asarray(first.line.coords[0]), np.asarray(first.line.coords[-1])
            b0, b1 = np.asarray(second.line.coords[0]), np.asarray(second.line.coords[-1])
            da, db = a1 - a0, b1 - b0
            cross = float(_cross2(da, db))
            norms = float(np.linalg.norm(da) * np.linalg.norm(db))
            if norms <= 1e-12 or abs(cross) / norms < minimum_sine:
                continue
            t = float(_cross2(b0 - a0, db) / cross)
            intersection = a0 + da * t
            if min(np.linalg.norm(intersection - point) for point in (a0, a1)) <= distance and min(np.linalg.norm(intersection - point) for point in (b0, b1)) <= distance:
                first.line = LineString([intersection if np.linalg.norm(intersection - a0) <= np.linalg.norm(intersection - a1) else a0, intersection if np.linalg.norm(intersection - a1) < np.linalg.norm(intersection - a0) else a1])
                second.line = LineString([intersection if np.linalg.norm(intersection - b0) <= np.linalg.norm(intersection - b1) else b0, intersection if np.linalg.norm(intersection - b1) < np.linalg.norm(intersection - b0) else b1])
    # A fitted run can terminate a few centimetres before a perpendicular run
    # (T-junction).  Move such an endpoint ALONG its own face onto the
    # perpendicular face, so the face keeps its measured direction.
    tolerance = 0.02 * distance / 0.30
    for face in lines:
        if face.line is None:
            continue
        p0, p1 = np.asarray(face.line.coords[0], dtype=float), np.asarray(face.line.coords[-1], dtype=float)
        direction = p1 - p0
        length = float(np.linalg.norm(direction))
        if length <= 1e-12:
            continue
        coordinates = [p0, p1]
        for endpoint_index, endpoint in enumerate((p0, p1)):
            candidates: list[tuple[float, np.ndarray]] = []
            for other in lines:
                if other is face or other.line is None:
                    continue
                q0, q1 = np.asarray(other.line.coords[0], dtype=float), np.asarray(other.line.coords[-1], dtype=float)
                other_direction = q1 - q0
                other_length = float(np.linalg.norm(other_direction))
                if other_length <= 1e-12:
                    continue
                cross = float(_cross2(direction, other_direction))
                if abs(cross) / (length * other_length) < minimum_sine:
                    continue
                t = float(_cross2(q0 - p0, other_direction) / cross)
                s = float(_cross2(q0 - p0, direction) / cross)
                if s * other_length < -tolerance or s * other_length > other_length + tolerance:
                    continue
                intersection = p0 + direction * t
                delta = float(np.linalg.norm(intersection - endpoint))
                # Never shrink a face to (nearly) nothing.
                remaining = coordinates[1 - endpoint_index]
                if delta <= distance and float(np.linalg.norm(intersection - remaining)) >= 0.5 * length:
                    candidates.append((delta, intersection))
            if candidates:
                coordinates[endpoint_index] = min(candidates, key=lambda item: item[0])[1]
        face.line = LineString(coordinates)


def _face_segment(face: WallFace, start: float, end: float) -> WallFace:
    assert face.line is not None
    segment = LineString([face.line.interpolate(start), face.line.interpolate(end)])
    return WallFace("line", face.points, line=segment, rms=face.rms, max_residual=face.max_residual, axis_snapped=face.axis_snapped)


def _cross2(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """Return the signed 2D cross product without NumPy's deprecated shortcut."""
    return np.asarray(first)[..., 0] * np.asarray(second)[..., 1] - np.asarray(first)[..., 1] * np.asarray(second)[..., 0]
