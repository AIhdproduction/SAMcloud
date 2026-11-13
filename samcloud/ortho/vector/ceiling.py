"""Ceiling-supported room polygons and conservative wall-gap completion."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

import cv2
import numpy as np
from scipy import ndimage
from shapely.geometry import LineString, Point, Polygon
from skimage.segmentation import watershed

from .footprints import _regularised, wall_axis_rectangle
from .raster_grid import RasterGrid, mask_to_polygons
from .regularize import regularize_polygon
from .wall_lines import WallFace, line_is_doorway


@dataclass(frozen=True, slots=True)
class CeilingSettings:
    close_radius_m: float = 0.15
    marker_distance_m: float = 0.25
    neck_width_m: float = 0.50
    min_room_area_m2: float = 1.0
    edge_min_length_m: float = 0.30
    cover_distance_m: float = 0.10
    duplicate_distance_m: float = 0.25
    axis_tolerance_deg: float = 5.0
    sliver_width_m: float = 0.20
    hole_fraction: float = 0.25
    hole_owner_share: float = 0.6


@dataclass(frozen=True, slots=True)
class CeilingRooms:
    polygons: list[Polygon]
    mask: np.ndarray
    coverage_mask: np.ndarray
    skipped: int


def ceiling_rooms(ceiling_count: np.ndarray, soffit: np.ndarray, wall_evidence: np.ndarray, grid: RasterGrid, theta: float, units_per_metre: float, settings: CeilingSettings = CeilingSettings()) -> CeilingRooms:
    """Extract room polygons from a coarse, dense ceiling-coverage raster."""
    ceiling = np.asarray(ceiling_count) > 0
    coarse_size = 0.05 * units_per_metre
    factor = max(1, int(round(coarse_size / grid.pixel_size)))
    coarse = _coarse_occupied(ceiling, factor)
    radius = max(1, int(round(0.20 * units_per_metre / (factor * grid.pixel_size))))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
    coarse = cv2.morphologyEx(coarse.astype(np.uint8), cv2.MORPH_CLOSE, kernel).astype(bool)
    coarse = _fill_small_holes(coarse, factor * grid.pixel_size, 1.0 * units_per_metre**2)
    coverage_mask = cv2.resize(coarse.astype(np.uint8), (grid.width, grid.height), interpolation=cv2.INTER_NEAREST).astype(bool)
    coverage_mask = coverage_mask[:grid.height, :grid.width]
    separator = cv2.dilate((np.asarray(soffit, dtype=bool) | np.asarray(wall_evidence, dtype=bool)).astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    mask = coverage_mask & ~separator
    distance = ndimage.distance_transform_edt(mask) * grid.pixel_size
    markers, _ = ndimage.label(distance >= settings.marker_distance_m * units_per_metre)
    if markers.max() == 0 and mask.any():
        markers, _ = ndimage.label(mask)
    labels = watershed(-distance, markers, mask=mask) if markers.max() else np.zeros(mask.shape, dtype=np.int32)
    labels = _fill_ceiling_holes(labels, coverage_mask, grid, theta, units_per_metre, settings)
    polygons: list[Polygon] = []
    skipped = 0
    for identifier in range(1, int(labels.max()) + 1):
        region = labels == identifier
        if float(region.sum()) * grid.pixel_size**2 < settings.min_room_area_m2 * units_per_metre**2:
            continue
        polygon = _room_polygon(region, grid, theta, units_per_metre)
        if polygon is None:
            # A strip leaking through a gap in the separator collapses to a
            # zero-width spur and self-intersects the outline; retry without it.
            trimmed = _without_slivers(region, grid.pixel_size, settings.sliver_width_m * units_per_metre)
            polygon = _room_polygon(trimmed, grid, theta, units_per_metre) if trimmed.any() else None
        if polygon is None:
            skipped += 1
            continue
        polygons.append(polygon)
    return CeilingRooms(polygons, mask, coverage_mask, skipped)


def ceiling_wall_pieces(rooms: Iterable[Polygon], measured_faces: Iterable[WallFace], free_floor: np.ndarray, ceiling_count: np.ndarray, ceiling_z: np.ndarray, soffit: np.ndarray, wall_evidence: np.ndarray, grid: RasterGrid, height: float, units_per_metre: float, settings: CeilingSettings = CeilingSettings(), furniture: Iterable[Polygon] = (), theta: float | None = None) -> tuple[list[WallFace], dict[str, int]]:
    """Fill only plausible uncovered room-edge runs with ceiling-supported walls."""
    measured = [face for face in measured_faces if face.kind == "line" and face.line is not None]
    lines = [face.line for face in measured if face.line is not None]
    stats = {"floor_continues": 0, "ceiling_step": 0, "doorway": 0, "short": 0, "not_manhattan": 0, "floor_both_sides": 0, "furniture_cross": 0, "duplicate_parallel": 0}
    furnishings = list(furniture)
    room_list = list(rooms)
    axis_theta = _infer_axis_theta(room_list) if theta is None else theta
    additions: list[WallFace] = []
    for room in room_list:
        coordinates = list(room.exterior.coords)
        for first, second in zip(coordinates, coordinates[1:], strict=False):
            edge = LineString((first, second))
            if edge.length < settings.edge_min_length_m * units_per_metre:
                continue
            covered = _covered_samples(edge, lines, settings.cover_distance_m * units_per_metre, settings.axis_tolerance_deg)
            parallel_covered = _parallel_overlap_samples(edge, lines, settings.duplicate_distance_m * units_per_metre, settings.axis_tolerance_deg)
            if bool((parallel_covered & ~covered).any()):
                stats["duplicate_parallel"] += 1
                covered = covered | parallel_covered
            for candidate in _uncovered_runs(edge, covered, 0.02 * units_per_metre):
                if candidate.length < settings.edge_min_length_m * units_per_metre:
                    stats["short"] += 1
                    continue
                if not _line_manhattan(candidate, axis_theta, 1.0):
                    stats["not_manhattan"] += 1
                    continue
                normal = _outer_normal(candidate, room)
                if any(candidate.intersection(item.buffer(0.01 * units_per_metre)).length > 0.10 * candidate.length for item in furnishings):
                    stats["furniture_cross"] += 1
                    continue
                if _both_sides_fraction(free_floor, grid, candidate, normal, 0.04 * units_per_metre) >= 0.50:
                    stats["floor_both_sides"] += 1
                    if _outside_fraction(free_floor, grid, candidate, normal, 0.30 * units_per_metre) >= 0.50:
                        stats["floor_continues"] += 1
                    continue
                if _outside_fraction(free_floor, grid, candidate, normal, 0.30 * units_per_metre) >= 0.50:
                    stats["floor_continues"] += 1
                    continue
                if _ceiling_continues(candidate, normal, ceiling_count, ceiling_z, grid, height, units_per_metre):
                    stats["ceiling_step"] += 1
                    continue
                if line_is_doorway(candidate, normal, units_per_metre, grid, free_floor):
                    stats["doorway"] += 1
                    continue
                snapped = _snap_to_measured_offset(candidate, lines, settings.duplicate_distance_m * units_per_metre)
                additions.append(WallFace("line", np.empty((0, 2), dtype=float), line=snapped, axis_snapped=True, source="ceiling"))
    deduplicated = _deduplicate_lines(additions, measured, units_per_metre)
    deduplicated, removed = _deduplicate_parallel_pieces(deduplicated, lines, settings.duplicate_distance_m * units_per_metre, settings.axis_tolerance_deg)
    stats["duplicate_parallel"] += removed
    return deduplicated, stats


def snap_exterior_faces_to_flucht(faces: list[WallFace], free_floor: np.ndarray, grid: RasterGrid, units_per_metre: float) -> None:
    """Put physically joined parallel exterior faces on the longest face's Flucht.

    Ceiling completion can discover a separated portion of an already measured
    exterior wall.  The two portions need not overlap along the wall direction,
    so ordinary Shapely line distance is not sufficient here.  A floor strip
    between the infinite face lines is the conservative separator: it preserves
    genuinely distinct parallel walls while merging two observations of one
    solid wall.
    """
    candidates = [face for face in faces if face.kind == "line" and face.line is not None and face.line.length > 1e-9]
    neighbours: dict[int, set[int]] = {index: set() for index in range(len(candidates))}
    limit = 0.25 * units_per_metre
    for first_index, first in enumerate(candidates):
        for second_index in range(first_index + 1, len(candidates)):
            second = candidates[second_index]
            if not _parallel(_direction(first.line), _direction(second.line), 2.0):
                continue
            normal = np.asarray((-_direction(first.line)[1], _direction(first.line)[0]))
            offset = abs(float(np.dot(np.asarray(second.line.coords[0]) - np.asarray(first.line.coords[0]), normal)))
            if offset > limit or _floor_separates_faces(first.line, second.line, free_floor, grid):
                continue
            neighbours[first_index].add(second_index)
            neighbours[second_index].add(first_index)
    visited: set[int] = set()
    for seed in range(len(candidates)):
        if seed in visited:
            continue
        pending, group = [seed], []
        visited.add(seed)
        while pending:
            index = pending.pop()
            group.append(index)
            for other in neighbours[index]:
                if other not in visited:
                    visited.add(other)
                    pending.append(other)
        if len(group) < 2:
            continue
        reference = max((candidates[index] for index in group), key=lambda face: face.line.length)
        origin = np.asarray(reference.line.coords[0], dtype=float)
        axis = _direction(reference.line)
        for index in group:
            face = candidates[index]
            values = [float(np.dot(np.asarray(point, dtype=float) - origin, axis)) for point in face.line.coords]
            face.line = LineString([tuple(origin + axis * value) for value in values])


def _floor_separates_faces(first: LineString, second: LineString, free_floor: np.ndarray, grid: RasterGrid) -> bool:
    """Return whether free floor occupies the strip between two infinite faces."""
    mask = np.asarray(free_floor, dtype=bool)
    if mask.shape != (grid.height, grid.width):
        return False
    probe, target = (first, second) if first.length <= second.length else (second, first)
    target_origin = np.asarray(target.coords[0], dtype=float)
    target_axis = _direction(target)
    values: list[bool] = []
    for point in _samples(probe, max(0.02, probe.length / 6.0)):
        source = np.asarray((point.x, point.y), dtype=float)
        projected = target_origin + target_axis * float(np.dot(source - target_origin, target_axis))
        distance = float(np.linalg.norm(projected - source))
        if distance <= grid.pixel_size:
            continue
        strip = [source + (projected - source) * fraction for fraction in np.linspace(0.20, 0.80, 4)]
        values.append(any(_mask_value(mask, grid, float(item[0]), float(item[1])) for item in strip))
    return bool(values) and float(np.mean(values)) >= 0.50


def _trim_edges(polygon: Polygon, minimum: float) -> Polygon | None:
    coordinates = list(polygon.exterior.coords[:-1])
    if len(coordinates) < 4:
        return None
    kept = [point for index, point in enumerate(coordinates) if math.dist(point, coordinates[index - 1]) >= minimum]
    candidate = Polygon(kept)
    return candidate if len(kept) >= 4 and candidate.is_valid and candidate.area > 0.0 else polygon


def _fill_ceiling_holes(labels: np.ndarray, coverage: np.ndarray, grid: RasterGrid, theta: float, units_per_metre: float, settings: CeilingSettings) -> np.ndarray:
    """Give enclosed ceiling holes (lamps, thinned ceiling) back to the room that owns them.

    A hole whose rim belongs mostly to one room and that is small compared with that room would otherwise
    cut a notch into the room outline where it touches the wall separator.  The hole is only filled inside
    the room's own axis-aligned bounding box, so it never carries a room across its wall.
    """
    count, holes, stats, _ = cv2.connectedComponentsWithStats((~coverage).astype(np.uint8), connectivity=8)
    if count <= 1 or labels.max() == 0:
        return labels
    output = labels.copy()
    reach = max(1, int(round(0.05 * units_per_metre / grid.pixel_size)))
    kernel = np.ones((2 * reach + 1, 2 * reach + 1), np.uint8)
    cosine, sine = math.cos(theta), math.sin(theta)
    for identifier in range(1, count):
        left, top, width, height, area = stats[identifier]
        if left == 0 or top == 0 or left + width >= coverage.shape[1] or top + height >= coverage.shape[0]:
            continue
        window = (slice(max(top - 2 * reach, 0), top + height + 2 * reach), slice(max(left - 2 * reach, 0), left + width + 2 * reach))
        hole = holes[window] == identifier
        rim = cv2.dilate(hole.astype(np.uint8), kernel).astype(bool) & ~hole
        owners = labels[window][rim]
        owners = owners[owners > 0]
        if not len(owners):
            continue
        values, counts = np.unique(owners, return_counts=True)
        owner = int(values[int(np.argmax(counts))])
        if counts.max() < settings.hole_owner_share * len(owners):
            continue
        region = labels == owner
        if float(area) > settings.hole_fraction * float(region.sum()):
            continue
        core = _without_slivers(region, grid.pixel_size, settings.sliver_width_m * units_per_metre)
        rows, columns = np.nonzero(core if core.any() else region)
        x = grid.min_x + (columns + 0.5) * grid.pixel_size
        y = grid.max_y - (rows + 0.5) * grid.pixel_size
        along, across = x * cosine + y * sine, -x * sine + y * cosine
        hole_rows, hole_columns = np.nonzero(hole)
        hole_x = grid.min_x + (hole_columns + window[1].start + 0.5) * grid.pixel_size
        hole_y = grid.max_y - (hole_rows + window[0].start + 0.5) * grid.pixel_size
        hole_along, hole_across = hole_x * cosine + hole_y * sine, -hole_x * sine + hole_y * cosine
        inside = (hole_along >= along.min()) & (hole_along <= along.max()) & (hole_across >= across.min()) & (hole_across <= across.max())
        output[hole_rows[inside] + window[0].start, hole_columns[inside] + window[1].start] = owner
    return output


def _room_polygon(region: np.ndarray, grid: RasterGrid, theta: float, units_per_metre: float) -> Polygon | None:
    polygon = _strict_manhattan_polygon(region, grid, theta, units_per_metre)
    if polygon is None or not polygon.is_valid or polygon.is_empty or polygon.area <= 0.0 or not _is_manhattan(polygon, theta, 0.5):
        return None
    return polygon


def _without_slivers(region: np.ndarray, pixel_size: float, width: float) -> np.ndarray:
    """Drop strips narrower than ``width`` that leak out of a room through a gap in its separator.

    A leaked strip collapses to a zero-width spur while the outline is being
    simplified, which makes the whole room outline self-intersecting and drops the
    room.  An opening with a square kernel keeps axis-aligned room corners exact;
    the largest remaining component is the room.
    """
    size = max(1, int(round(width / pixel_size)))
    if size <= 1:
        return region
    opened = cv2.morphologyEx(region.astype(np.uint8), cv2.MORPH_OPEN, np.ones((size, size), np.uint8))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(opened, connectivity=8)
    if count <= 1:
        return np.zeros_like(region, dtype=bool)
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return labels == largest


def _coarse_occupied(mask: np.ndarray, factor: int) -> np.ndarray:
    height = int(math.ceil(mask.shape[0] / factor))
    width = int(math.ceil(mask.shape[1] / factor))
    padded = np.pad(mask, ((0, height * factor - mask.shape[0]), (0, width * factor - mask.shape[1])), mode="constant")
    return padded.reshape(height, factor, width, factor).any(axis=(1, 3))


def _fill_small_holes(mask: np.ndarray, pixel: float, maximum_area: float, relative_area: float = 0.0) -> np.ndarray:
    """Fill enclosed holes in the ceiling coverage.

    A hole is filled when it is smaller than ``maximum_area`` or smaller than ``relative_area`` times the
    coverage that surrounds it: a ceiling thinned around lamps leaves a hole that is large in absolute
    terms but small compared with its room, and its edge would otherwise cut a notch into the room outline.
    """
    inverse = ~mask
    count, labels, stats, _ = cv2.connectedComponentsWithStats(inverse.astype(np.uint8), connectivity=8)
    covered_count, covered_labels, covered_stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    output = mask.copy()
    limit = maximum_area / max(pixel**2, 1e-9)
    for identifier in range(1, count):
        left, top, width, height, area = stats[identifier]
        touches_border = left == 0 or top == 0 or left + width >= mask.shape[1] or top + height >= mask.shape[0]
        if touches_border:
            continue
        allowed = limit
        if relative_area > 0.0:
            window = (slice(max(top - 1, 0), top + height + 1), slice(max(left - 1, 0), left + width + 1))
            around = np.unique(covered_labels[window][(labels[window] != identifier) & (covered_labels[window] > 0)])
            if len(around):
                surrounding = max(int(covered_stats[value, cv2.CC_STAT_AREA]) for value in around)
                allowed = max(allowed, relative_area * surrounding)
        if area <= allowed:
            output[labels == identifier] = True
    return output


def _strict_manhattan_polygon(mask: np.ndarray, grid: RasterGrid, theta: float, units_per_metre: float) -> Polygon | None:
    """Rotate the room raster, remove short staircase steps, and rotate it back."""
    angle = -math.degrees(theta)
    centre = ((mask.shape[1] - 1) / 2.0, (mask.shape[0] - 1) / 2.0)
    forward = cv2.getRotationMatrix2D(centre, angle, 1.0)
    rotated = cv2.warpAffine(mask.astype(np.uint8), forward, (mask.shape[1], mask.shape[0]), flags=cv2.INTER_NEAREST, borderValue=0)
    kernel_size = max(1, int(round(0.10 * units_per_metre / grid.pixel_size)))
    rotated = cv2.morphologyEx(rotated, cv2.MORPH_CLOSE, np.ones((kernel_size, kernel_size), np.uint8))
    contours, _ = cv2.findContours(rotated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    contour = cv2.approxPolyDP(contour, 0.10 * units_per_metre / grid.pixel_size, True).reshape((-1, 2)).astype(float)
    contour = _snap_axis_contour(contour)
    contour = _remove_short_steps(contour, 0.20 * units_per_metre / grid.pixel_size, 0.15 * units_per_metre**2 / grid.pixel_size**2)
    if len(contour) < 4:
        return None
    inverse = cv2.invertAffineTransform(forward)
    restored = cv2.transform(contour.reshape((-1, 1, 2)).astype(np.float32), inverse).reshape((-1, 2))
    points = [grid.pixel_to_local(float(row), float(column)) for column, row in restored]
    polygon = Polygon(points)
    return polygon if polygon.is_valid and polygon.area > 0.0 else None


def _snap_axis_contour(points: np.ndarray) -> np.ndarray:
    """Intersect alternating horizontal and vertical contour lines exactly."""
    if len(points) < 4:
        return points
    vertices = [np.asarray(point, dtype=float) for point in points]
    while len(vertices) > 4:
        directions = [abs((vertices[(index + 1) % len(vertices)] - point)[0]) >= abs((vertices[(index + 1) % len(vertices)] - point)[1]) for index, point in enumerate(vertices)]
        duplicate = next((index for index in range(len(directions)) if directions[index] == directions[(index + 1) % len(directions)]), None)
        if duplicate is None:
            break
        vertices.pop((duplicate + 1) % len(vertices))
    points = np.asarray(vertices, dtype=float)
    directions = []
    coordinates = []
    for first, second in zip(points, np.roll(points, -1, axis=0), strict=False):
        delta = second - first
        horizontal = abs(delta[0]) >= abs(delta[1])
        directions.append(horizontal)
        coordinates.append(float((first[1] + second[1]) / 2.0) if horizontal else float((first[0] + second[0]) / 2.0))
    if any(directions[index] == directions[(index + 1) % len(directions)] for index in range(len(directions))):
        return points
    output = []
    for index in range(len(points)):
        incoming = (index - 1) % len(points)
        if directions[incoming]:
            output.append((coordinates[index], coordinates[incoming]))
        else:
            output.append((coordinates[incoming], coordinates[index]))
    return np.asarray(output, dtype=float)


def _remove_short_steps(points: np.ndarray, minimum: float, maximum_area: float) -> np.ndarray:
    output = [np.asarray(point, dtype=float) for point in points]
    changed = True
    iterations = 0
    maximum_iterations = max(1, len(output) * 4)
    while changed and len(output) > 4 and iterations < maximum_iterations:
        changed = False
        for index, point in enumerate(output):
            previous = output[index - 1]
            following = output[(index + 1) % len(output)]
            first = float(np.linalg.norm(point - previous))
            second = float(np.linalg.norm(following - point))
            if min(first, second) >= minimum:
                continue
            triangle = abs(float(np.cross(point - previous, following - point))) / 2.0
            if triangle <= maximum_area:
                short = point - previous if first <= second else following - point
                if abs(short[0]) >= abs(short[1]):
                    replacement = np.asarray((previous[0], following[1]))
                else:
                    replacement = np.asarray((following[0], previous[1]))
                if np.allclose(replacement, point):
                    continue
                output[index] = replacement
                changed = True
                iterations += 1
                break
    return np.asarray(output, dtype=float)


def _is_manhattan(polygon: Polygon, theta: float, tolerance_deg: float) -> bool:
    axes = (np.asarray((math.cos(theta), math.sin(theta))), np.asarray((-math.sin(theta), math.cos(theta))))
    for first, second in zip(polygon.exterior.coords, list(polygon.exterior.coords)[1:], strict=False):
        direction = np.asarray(second) - np.asarray(first)
        length = float(np.linalg.norm(direction))
        if length <= 1e-9:
            continue
        direction /= length
        if min(abs(math.degrees(math.asin(float(np.clip(abs(np.cross(direction, axis)), 0.0, 1.0))))) for axis in axes) > tolerance_deg:
            return False
    return True


def _covered_samples(edge: LineString, faces: list[LineString], distance: float, tolerance_deg: float) -> np.ndarray:
    samples = _samples(edge, max(0.01, 0.02))
    direction = _direction(edge)
    result = []
    for point in samples:
        result.append(any(face.distance(point) <= distance and _parallel(direction, _direction(face), tolerance_deg) for face in faces))
    return np.asarray(result, dtype=bool)


def _parallel_overlap_samples(edge: LineString, faces: list[LineString], distance: float, tolerance_deg: float) -> np.ndarray:
    """Samples of the edge lying within `distance` of a parallel face and alongside its extent."""
    direction = _direction(edge)
    samples = _samples(edge, 0.02)
    result = np.zeros(len(samples), dtype=bool)
    for face in faces:
        face_direction = _direction(face)
        if not _parallel(direction, face_direction, tolerance_deg):
            continue
        origin = np.asarray(face.coords[0], dtype=float)
        normal = np.asarray((-face_direction[1], face_direction[0]))
        for index, point in enumerate(samples):
            offset = np.asarray((point.x, point.y)) - origin
            along = float(np.dot(offset, face_direction))
            if abs(float(np.dot(offset, normal))) <= distance and -1e-9 <= along <= face.length + 1e-9:
                result[index] = True
    return result


def _deduplicate_parallel_pieces(pieces: list[WallFace], measured_lines: list[LineString], distance: float, tolerance_deg: float) -> tuple[list[WallFace], int]:
    """Drop ceiling pieces that double another ceiling piece; keep the one nearer measured walls, then the longer."""
    def key(face: WallFace) -> tuple[float, float]:
        near = min((face.line.distance(line) for line in measured_lines), default=float("inf"))
        return (near, -face.line.length)

    kept: list[WallFace] = []
    removed = 0
    for face in sorted(pieces, key=key):
        line = face.line
        duplicate = False
        for other in kept:
            if _parallel(_direction(line), _direction(other.line), tolerance_deg) and _overlaps_parallel(line, other.line, distance):
                duplicate = True
                break
        if duplicate:
            removed += 1
        else:
            kept.append(face)
    order = {id(face): index for index, face in enumerate(pieces)}
    kept.sort(key=lambda face: order[id(face)])
    return kept, removed


def _overlaps_parallel(first: LineString, second: LineString, distance: float) -> bool:
    direction = _direction(second)
    normal = np.asarray((-direction[1], direction[0]))
    origin = np.asarray(second.coords[0], dtype=float)
    offsets = [np.asarray(point, dtype=float) - origin for point in first.coords]
    if max(abs(float(np.dot(offset, normal))) for offset in offsets) > distance:
        return False
    along = [float(np.dot(offset, direction)) for offset in offsets]
    return min(max(along), second.length) - max(min(along), 0.0) > 1e-6


def _uncovered_runs(edge: LineString, covered: np.ndarray, spacing: float) -> list[LineString]:
    output: list[LineString] = []
    start: int | None = None
    for index, value in enumerate(covered.tolist() + [True]):
        if not value and start is None:
            start = index
        elif value and start is not None:
            a = edge.interpolate(edge.length * start / max(len(covered) - 1, 1))
            b = edge.interpolate(edge.length * (index - 1) / max(len(covered) - 1, 1))
            if a.distance(b) > 1e-9:
                output.append(LineString((a, b)))
            start = None
    return output


def _outer_normal(line: LineString, room: Polygon) -> np.ndarray:
    direction = _direction(line)
    normal = np.asarray((-direction[1], direction[0]))
    return -normal if room.exterior.is_ccw else normal


def _outside_fraction(mask: np.ndarray, grid: RasterGrid, line: LineString, normal: np.ndarray, offset: float) -> float:
    samples = _samples(line, 0.02)
    return float(np.mean([_mask_value(mask, grid, point.x + normal[0] * offset, point.y + normal[1] * offset) for point in samples])) if samples else 0.0


def _both_sides_fraction(mask: np.ndarray, grid: RasterGrid, line: LineString, normal: np.ndarray, offset: float) -> float:
    samples = _samples(line, 0.02)
    values = [_mask_value(mask, grid, point.x + normal[0] * offset, point.y + normal[1] * offset) and _mask_value(mask, grid, point.x - normal[0] * offset, point.y - normal[1] * offset) for point in samples]
    return float(np.mean(values)) if values else 0.0


def _line_manhattan(line: LineString, theta: float, tolerance_deg: float) -> bool:
    direction = _direction(line)
    axes = (np.asarray((math.cos(theta), math.sin(theta))), np.asarray((-math.sin(theta), math.cos(theta))))
    return min(abs(math.degrees(math.asin(float(np.clip(abs(np.cross(direction, axis)), 0.0, 1.0))))) for axis in axes) <= tolerance_deg


def _infer_axis_theta(rooms: Iterable[Polygon]) -> float:
    edges = [LineString((first, second)) for room in rooms for first, second in zip(room.exterior.coords, list(room.exterior.coords)[1:], strict=False)]
    if not edges:
        return 0.0
    edge = max(edges, key=lambda value: value.length)
    direction = _direction(edge)
    return math.atan2(direction[1], direction[0]) % (math.pi / 2.0)


def _ceiling_continues(line: LineString, normal: np.ndarray, count: np.ndarray, z: np.ndarray, grid: RasterGrid, height: float, units_per_metre: float) -> bool:
    values: list[bool] = []
    for point in _samples(line, 0.02):
        inside = _mask_value(count > 0, grid, point.x, point.y)
        current = _raster_value(z, grid, point.x, point.y)
        outer = []
        for offset in np.linspace(0.10 * units_per_metre, 0.30 * units_per_metre, 3):
            value = _raster_value(z, grid, point.x + normal[0] * offset, point.y + normal[1] * offset)
            outer.append(value > 0.0 and (not inside or abs(value - current) <= 0.12 * height))
        values.append(all(outer))
    return bool(values) and float(np.mean(values)) >= 0.50


def _snap_to_measured_offset(line: LineString, faces: list[LineString], distance: float) -> LineString:
    """Move a ceiling edge to a nearby measured infinite wall alignment.

    Shapely's line distance includes end separation, which is unsuitable for
    separate pieces of the same Flucht.  Compare perpendicular offsets instead.
    """
    direction = _direction(line)
    normal = np.asarray((-direction[1], direction[0]))
    midpoint = np.asarray(line.interpolate(0.5, normalized=True).coords[0])
    parallels = []
    for face in faces:
        if not _parallel(direction, _direction(face), 2.0):
            continue
        origin = np.asarray(face.coords[0], dtype=float)
        offset = float(np.dot(origin - midpoint, normal))
        if abs(offset) <= distance:
            parallels.append(offset)
    if not parallels:
        return line
    shift = normal * float(np.median(parallels))
    return LineString([tuple(np.asarray(point) + shift) for point in line.coords])


def _deduplicate_lines(additions: list[WallFace], measured: list[WallFace], units_per_metre: float) -> list[WallFace]:
    output: list[WallFace] = []
    existing = [face.line for face in measured if face.line is not None]
    for face in additions:
        if face.line is None or any(_parallel(_direction(face.line), _direction(line), 5.0) and _overlaps_parallel(face.line, line, 0.02 * units_per_metre) for line in existing):
            continue
        if not any(_parallel(_direction(face.line), _direction(other.line), 5.0) and _overlaps_parallel(face.line, other.line, 0.02 * units_per_metre) for other in output if other.line is not None):
            output.append(face)
    return output


def _samples(line: LineString, spacing: float) -> list[Point]:
    return [line.interpolate(value) for value in np.linspace(0.0, line.length, max(2, int(math.ceil(line.length / spacing)) + 1))]


def _direction(line: LineString) -> np.ndarray:
    vector = np.asarray(line.coords[-1]) - np.asarray(line.coords[0])
    return vector / max(float(np.linalg.norm(vector)), 1e-12)


def _parallel(first: np.ndarray, second: np.ndarray, tolerance_deg: float) -> bool:
    return abs(math.degrees(math.asin(float(np.clip(abs(np.cross(first, second)), 0.0, 1.0))))) <= tolerance_deg


def _mask_value(mask: np.ndarray, grid: RasterGrid, x: float, y: float) -> bool:
    row, col = grid.local_to_pixel(x, y)
    row, col = int(round(row)), int(round(col))
    return 0 <= row < mask.shape[0] and 0 <= col < mask.shape[1] and bool(mask[row, col])


def _raster_value(values: np.ndarray, grid: RasterGrid, x: float, y: float) -> float:
    row, col = grid.local_to_pixel(x, y)
    row, col = int(round(row)), int(round(col))
    return float(values[row, col]) if 0 <= row < values.shape[0] and 0 <= col < values.shape[1] else -1.0
