"""Vertical-surface evidence and conservative built-in extraction for indoor plans."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

import cv2
import numpy as np
from shapely.geometry import LineString, Point, Polygon
from shapely.ops import unary_union

from .raster_grid import RasterGrid, mask_to_polygons


@dataclass(frozen=True, slots=True)
class DoorLeaf:
    """Observed open leaf geometry in plan coordinates."""

    angle_deg: float
    hinge_side: str
    swing_side: int
    mode: str = "open"
    direction_angle_deg: float | None = None


@dataclass(frozen=True, slots=True)
class VerticalEvidence:
    """Floor-connected vertical occupancy derived from one streamed point cloud."""

    full: np.ndarray
    counter: np.ndarray
    top_height: np.ndarray
    bitmask: np.ndarray
    ceiling_count: np.ndarray | None = None
    ceiling_z: np.ndarray | None = None
    soffit: np.ndarray | None = None
    wall_bits: np.ndarray | None = None


@dataclass(frozen=True, slots=True)
class VerticalSettings:
    bins: int = 16
    floor_fraction: float = 0.04
    ceiling_fraction: float = 0.97
    fill_fraction: float = 0.70
    full_fraction: float = 0.85
    counter_fraction: float = 0.30
    wall_band_m: float = 0.10
    min_length_m: float = 0.40
    axis_tolerance_deg: float = 5.0
    rms_m: float = 0.02
    support_fraction: float = 0.70
    max_gap_m: float = 0.15
    merge_offset_m: float = 0.03
    furniture_distance_m: float = 0.05
    isolation_distance_m: float = 0.60


def vertical_evidence(bitmask: np.ndarray, height: float, settings: VerticalSettings = VerticalSettings()) -> VerticalEvidence:
    """Apply the prototype's 3x3 OR, floor-contact, and solid-fill checks."""
    bits = np.asarray(bitmask, dtype=np.uint32)
    if bits.ndim != 2:
        raise ValueError("vertical bitmask must be a two-dimensional raster")
    if height <= 0.0:
        empty = np.zeros_like(bits, dtype=bool)
        return VerticalEvidence(empty, empty, np.full(bits.shape, -1.0, dtype=np.float32), bits)
    aggregate = _or_neighbourhood(bits)
    occupied = np.stack([((aggregate >> np.uint32(index)) & np.uint32(1)).astype(bool) for index in range(settings.bins)], axis=-1)
    any_occupied = occupied.any(axis=-1)
    top = np.where(any_occupied, settings.bins - 1 - np.argmax(occupied[..., ::-1], axis=-1), -1)
    lowest = np.where(any_occupied, np.argmax(occupied, axis=-1), settings.bins)
    cumulative = np.cumsum(occupied, axis=-1)
    filled = np.where(any_occupied, np.take_along_axis(cumulative, np.clip(top, 0, settings.bins - 1)[..., None], axis=-1)[..., 0] / np.maximum(top + 1, 1), 0.0)
    floor_connected = (lowest == 0) & (filled >= settings.fill_fraction)
    full = floor_connected & (top >= int(settings.full_fraction * settings.bins))
    counter = floor_connected & (top >= int(settings.counter_fraction * settings.bins)) & ~full
    top_height = np.where(top >= 0, (top.astype(np.float32) + 1.0) / settings.bins * float(height), -1.0).astype(np.float32)
    return VerticalEvidence(full, counter, top_height, aggregate)


def detect_door_leaf(opening: object, evidence: VerticalEvidence, grid: RasterGrid, height: float, units_per_metre: float) -> DoorLeaf | None:
    """Find an observed thin, floor-to-ceiling open door leaf near one jamb.

    The full vertical mask is deliberately used without a class filter: scans
    often classify a physical leaf as wall or unknown rather than class 103.
    """
    # A leaf is a thin vertical observation, not necessarily a solid cabinet
    # front.  Use the vertical-evidence top raster, then enforce the leaf's
    # own height, thickness, length and hinge constraints below.
    mask = np.asarray(evidence.top_height, dtype=float) >= 0.70 * height
    if not mask.any():
        mask = np.asarray(evidence.full, dtype=bool)
    if not mask.any() or height <= 0.0:
        return None
    centre = np.asarray(getattr(opening, "center"), dtype=float)
    width = float(getattr(opening, "width"))
    wall_angle = float(getattr(opening, "axis_angle"))
    axis = np.asarray((math.cos(wall_angle), math.sin(wall_angle)))
    normal = np.asarray((-axis[1], axis[0]))
    extent = 1.25 * width + 0.15 * units_per_metre
    corners = np.asarray([centre + sign_a * axis * extent + sign_n * normal * extent for sign_a in (-1.0, 1.0) for sign_n in (-1.0, 1.0)])
    pixels = np.asarray([grid.local_to_pixel(float(point[0]), float(point[1]))[::-1] for point in corners], dtype=int)
    c0, r0 = pixels.min(axis=0); c1, r1 = pixels.max(axis=0)
    r0, r1 = max(0, r0), min(grid.height - 1, r1)
    c0, c1 = max(0, c0), min(grid.width - 1, c1)
    if r1 <= r0 or c1 <= c0:
        return None
    roi = (mask[r0:r1 + 1, c0:c1 + 1].astype(np.uint8) * 255)
    raw = cv2.HoughLinesP(roi, 1, np.pi / 360.0, threshold=max(4, int(round(0.25 * units_per_metre / grid.pixel_size))), minLineLength=max(3, int(round(0.50 * width / grid.pixel_size))), maxLineGap=max(1, int(round(0.05 * units_per_metre / grid.pixel_size))) )
    if raw is None:
        return None
    # ``full`` includes the deliberate 3x3 aggregation used for robust line
    # detection.  Measure thickness from the original point occupancy instead,
    # otherwise a genuinely thin flat leaf is widened past the 7 cm limit.
    thickness_mask = np.asarray(evidence.bitmask, dtype=np.uint32) != 0
    if not thickness_mask.any():
        thickness_mask = mask
    distance_field = cv2.distanceTransform(thickness_mask.astype(np.uint8), cv2.DIST_L2, 3)
    best: tuple[float, DoorLeaf] | None = None
    for x0, y0, x1, y1 in raw[:, 0, :]:
        first = np.asarray(grid.pixel_to_local(float(y0 + r0), float(x0 + c0)))
        second = np.asarray(grid.pixel_to_local(float(y1 + r0), float(x1 + c0)))
        vector = second - first
        length = float(np.linalg.norm(vector))
        if not 0.60 * width <= length <= 1.15 * width:
            continue
        direction = vector / max(length, 1e-12)
        angle = math.degrees(math.asin(float(np.clip(abs(np.cross(direction, axis)), 0.0, 1.0))))
        samples = np.linspace(first, second, max(3, int(round(length / max(grid.pixel_size, 1e-9)))))
        sample_pixels = [grid.local_to_pixel(float(point[0]), float(point[1])) for point in samples]
        radii = [float(distance_field[int(round(row)), int(round(col))]) for row, col in sample_pixels if 0 <= int(round(row)) < grid.height and 0 <= int(round(col)) < grid.width]
        top_samples = [float(evidence.top_height[int(round(row)), int(round(col))]) for row, col in sample_pixels if 0 <= int(round(row)) < grid.height and 0 <= int(round(col)) < grid.width]
        # A line can meet a jamb or a wall return at one sampled pixel.  Use a
        # robust cross-section estimate so that local intersections do not turn
        # an otherwise thin observed leaf into a thick surface.
        thickness = 2.0 * float(np.percentile(radii, 75)) * grid.pixel_size if radii else float("inf")
        positive_tops = [value for value in top_samples if value > 0.0]
        if thickness > 0.07 * units_per_metre or (positive_tops and float(np.median(positive_tops)) < 0.70 * height):
            continue
        left = centre - axis * width / 2.0
        right = centre + axis * width / 2.0
        end, hinge, side = min(((first, left, "left"), (second, left, "left"), (first, right, "right"), (second, right, "right")), key=lambda item: float(np.linalg.norm(item[0] - item[1])))
        distance = float(np.linalg.norm(end - hinge))
        if distance > 0.12 * units_per_metre:
            continue
        other = second if np.allclose(end, first) else first
        leaf_direction = other - hinge
        swing_side = 1 if float(np.dot(leaf_direction, normal)) >= 0.0 else -1
        measured = math.degrees(math.atan2(float(leaf_direction[1]), float(leaf_direction[0])))
        offset = abs(float(np.dot((first + second) / 2.0 - centre, normal)))
        hinge_projection = float(np.dot(end - centre, axis))
        tip_projection = float(np.dot(other - centre, axis))
        midpoint_projection = (hinge_projection + tip_projection) / 2.0
        # The observed hinge can be up to 12 cm from the jamb.  Accept that
        # short overhang into the opening when the leaf's tip and midpoint are
        # outside its span; this matches flat leaves scanned just beside a jamb.
        outside_span = (
            (side == "left" and tip_projection <= -width / 2.0 and midpoint_projection <= -width / 2.0)
            or (side == "right" and tip_projection >= width / 2.0 and midpoint_projection >= width / 2.0)
        )
        is_flat = angle <= 15.0 and outside_span and 0.02 * units_per_metre <= offset <= 0.20 * units_per_metre
        if not is_flat and not 15.0 <= angle <= 165.0:
            continue
        if is_flat:
            swing_side = 1 if float(np.dot((first + second) / 2.0 - centre, normal)) >= 0.0 else -1
        candidate = DoorLeaf(180.0 if is_flat else measured, side, swing_side, "flat" if is_flat else "open", measured)
        score = distance + abs(length - width) * 0.10
        if best is None or score < best[0]:
            best = (score, candidate)
    return None if best is None else best[1]


def extract_builtins(
    evidence: VerticalEvidence,
    grid: RasterGrid,
    theta: float,
    wall_lines: Iterable[LineString],
    wall_evidence: np.ndarray,
    furniture: Iterable[Polygon],
    countertop: np.ndarray,
    units_per_metre: float,
    *,
    kitchen_evidence: np.ndarray | None = None,
    settings: VerticalSettings = VerticalSettings(),
) -> tuple[list[dict[str, object]], dict[str, int], list[Polygon]]:
    """Return conservative A-EINB entities from floor-standing vertical surfaces.

    Segments are intentionally accepted only when they have a straight, dense and
    wall-axis-aligned support.  This keeps vertical furniture fragments out of
    the architectural layer.
    """
    walls = list(wall_lines)
    candidate = (np.asarray(evidence.full, dtype=bool) | np.asarray(evidence.counter, dtype=bool)) & ~np.asarray(wall_evidence, dtype=bool)
    wall_mask = _line_mask(candidate.shape, grid, walls, settings.wall_band_m * units_per_metre)
    candidate &= ~wall_mask
    stats = {"furniture_edge": 0, "isolated": 0, "short": 0, "rms": 0, "support": 0}
    counter_top = np.asarray(countertop, dtype=bool)
    kitchen = np.asarray(kitchen_evidence, dtype=bool) if kitchen_evidence is not None else np.zeros_like(candidate)
    entities: list[dict[str, object]] = []
    polygons: list[Polygon] = []
    accepted: list[LineString] = []
    count, labels = cv2.connectedComponents(candidate.astype(np.uint8), connectivity=8)
    for identifier in range(1, count):
        component = labels == identifier
        for line in _component_lines(component, grid, theta, units_per_metre, settings):
            reason = _line_rejection(line, component, grid, walls, furniture, counter_top, accepted, units_per_metre, settings)
            if reason is not None:
                stats[reason] += 1
                continue
            counter_polygon = _counter_polygon(line, counter_top, grid, theta, walls, units_per_metre, settings)
            if counter_polygon is not None:
                polygons.append(counter_polygon)
                near_kitchen = kitchen.any() and _mask_polygon(kitchen, grid).distance(counter_polygon) <= 0.30 * units_per_metre
                height_samples = evidence.top_height[_polygon_pixels(counter_polygon, grid)]
                height = float(np.median(height_samples[height_samples > 0.0])) / max(units_per_metre, 1e-9) if np.any(height_samples > 0.0) else 0.0
                entities.append({"type": "polyline", "points": _points(counter_polygon), "closed": True, "holes": [], "_einb_kind": "counter", "_einb_id": len(polygons)})
                entities.append({"type": "text", "_einb_owner": len(polygons), "text": f"{'Küche' if near_kitchen else 'Einbau'} h={height:.2f}", "point": list(counter_polygon.representative_point().coords[0]), "height": 0.10 * units_per_metre, "rotation": 0.0})
            else:
                entities.append({"type": "polyline", "points": [[float(x), float(y)] for x, y in line.coords], "closed": False, "_einb_kind": "segment"})
                accepted.append(line)
    entities, polygons = _deduplicate(entities, polygons, settings.merge_offset_m * units_per_metre)
    return entities, stats, polygons


def filter_builtins_to_rooms(entities: list[dict[str, object]], room_polygons: Iterable[Polygon], wall_lines: Iterable[LineString], furniture: Iterable[Polygon], free_floor: np.ndarray, grid: RasterGrid, units_per_metre: float) -> tuple[list[dict[str, object]], list[Polygon], dict[str, int]]:
    """Reject built-ins outside rooms, on walls, over open floor, or duplicated by furniture."""
    rooms = list(room_polygons)
    room_union = unary_union(rooms) if rooms else Polygon()
    walls = list(wall_lines)
    furnishings = list(furniture)
    rejected = {"outside_rooms": 0, "wall_duplicate": 0, "covers_floor": 0, "furn_duplicate": 0}
    kept: list[dict[str, object]] = []
    polygons: list[Polygon] = []
    for entity in entities:
        if entity.get("type") != "polyline":
            kept.append(entity)
            continue
        points = entity.get("points", [])
        if len(points) < 2:
            continue
        if entity.get("closed"):
            polygon = Polygon(points)
            if polygon.is_empty or polygon.area <= 0.0 or polygon.intersection(room_union).area < 0.80 * polygon.area:
                rejected["outside_rooms"] += 1
                continue
            floor_mask = _polygon_pixels(polygon, grid)
            if floor_mask.any() and float(np.mean(np.asarray(free_floor, dtype=bool)[floor_mask])) > 0.20:
                rejected["covers_floor"] += 1
                continue
            overlap = max((polygon.intersection(item).area / max(min(polygon.area, item.area), 1e-9) for item in furnishings), default=0.0)
            touches_wall = any(polygon.distance(wall) <= 0.05 * units_per_metre for wall in walls)
            if overlap > 0.50 and not touches_wall:
                rejected["furn_duplicate"] += 1
                continue
            polygons.append(polygon)
            kept.append(entity)
            continue
        line = LineString(points)
        if line.intersection(room_union).length < 0.80 * line.length:
            rejected["outside_rooms"] += 1
            continue
        samples = _line_samples(line, 0.02 * units_per_metre)
        duplicate = max((float(np.mean([wall.distance(point) <= 0.10 * units_per_metre for point in samples])) for wall in walls), default=0.0)
        if duplicate >= 0.50:
            rejected["wall_duplicate"] += 1
            continue
        kept.append(entity)
    return kept, polygons, rejected


def tidy_builtins(
    entities: list[dict[str, object]],
    obstacle_texts: Iterable[dict[str, object]],
    units_per_metre: float,
    *,
    merge_gap_m: float = 0.10,
    min_label_area_m2: float = 0.30,
) -> tuple[list[dict[str, object]], dict[str, int]]:
    """Clean the A-EINB layer: one polygon and one label per built-in, no overlapping labels.

    * Closed polygons with the same label whose gap is at most ``merge_gap_m``
      merge into one polygon, regularised to a rectangle or a simple L-shape.
    * Texts without a polygon (their polygon was rejected) are dropped and each
      merged polygon keeps exactly one text.
    * Near-duplicate parallel segments collapse to the longest one; segments
      that only retrace a polygon edge are dropped.
    * A label that overlaps another label (also the fixed ``obstacle_texts``)
      moves inside its own polygon, or is dropped for a polygon below 0.3 m2.
    """
    upm = max(float(units_per_metre), 1e-9)
    stats = {"before": len(entities), "polygons_merged": 0, "texts_dropped": 0, "segments_dropped": 0, "labels_moved": 0}
    polygons = [item for item in entities if item.get("type") == "polyline" and item.get("closed") and len(item.get("points", [])) >= 3]
    segments = [item for item in entities if item.get("type") == "polyline" and not item.get("closed")]
    texts = [item for item in entities if item.get("type") == "text"]
    others = [item for item in entities if item not in polygons and item not in segments and item not in texts]
    shapes = [Polygon(item["points"]).buffer(0) for item in polygons]
    valid = [index for index, shape in enumerate(shapes) if isinstance(shape, Polygon) and shape.area > 0.0]
    owners: dict[int, list[dict[str, object]]] = {index: [] for index in valid}
    owner_ids = {polygons[index].get("_einb_id"): index for index in valid if polygons[index].get("_einb_id") is not None}
    for text in texts:
        if text.get("_einb_owner") is not None:
            # The counter text belongs to exactly one polygon; it is an orphan
            # once that polygon was rejected or de-duplicated.
            if text["_einb_owner"] in owner_ids:
                owners[owner_ids[text["_einb_owner"]]].append(text)
            else:
                stats["texts_dropped"] += 1
            continue
        point = Point(text["point"])
        candidates = [index for index in valid if shapes[index].distance(point) <= 0.05 * upm]
        if not candidates:
            stats["texts_dropped"] += 1
            continue
        owners[min(candidates, key=lambda index: shapes[index].area)].append(text)

    def base_label(index: int) -> str:
        for text in owners.get(index, []):
            return str(text.get("text", "")).split(" h=")[0].strip()
        return "Küche" if polygons[index].get("_einb_kind") == "fine_kitchen" else "Einbau"

    gap = merge_gap_m * upm
    parent = {index: index for index in valid}

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    for position, first in enumerate(valid):
        for second in valid[position + 1:]:
            if base_label(first) == base_label(second) and shapes[first].distance(shapes[second]) <= gap:
                parent[find(second)] = find(first)
    groups: dict[int, list[int]] = {}
    for index in valid:
        groups.setdefault(find(index), []).append(index)

    merged_polygons: list[dict[str, object]] = []
    merged_shapes: list[Polygon] = []
    labels: list[dict[str, object] | None] = []
    for members in groups.values():
        union = unary_union([shapes[index].buffer(gap / 2.0 + 1e-6, join_style=2) for index in members]).buffer(-(gap / 2.0 + 1e-6), join_style=2) if len(members) > 1 else shapes[members[0]]
        if not isinstance(union, Polygon):
            pieces = [piece for piece in getattr(union, "geoms", []) if isinstance(piece, Polygon)]
            if not pieces:
                continue
            union = max(pieces, key=lambda piece: piece.area)
        regular = _rectangle_or_ell(union, upm)
        stats["polygons_merged"] += len(members) - 1
        biggest = max(members, key=lambda index: shapes[index].area)
        first_entity = polygons[biggest]
        merged_polygons.append({**first_entity, "points": [[float(x), float(y)] for x, y in regular.exterior.coords[:-1]], "closed": True, "holes": []})
        merged_shapes.append(regular)
        member_texts = [text for index in members for text in owners.get(index, [])]
        with_height = [text for text in member_texts if " h=" in str(text.get("text", ""))]
        chosen = max(with_height or member_texts, key=lambda text: float(text.get("height", 0.0)), default=None) if member_texts else None
        stats["texts_dropped"] += max(0, len(member_texts) - (1 if chosen is not None else 0))
        labels.append(None if chosen is None else {**chosen, "point": list(regular.representative_point().coords[0])})

    boundary_union = unary_union([shape.boundary for shape in merged_shapes]) if merged_shapes else None
    kept_segments: list[dict[str, object]] = []
    for segment in sorted(segments, key=lambda item: LineString(item["points"]).length if len(item.get("points", [])) >= 2 else 0.0, reverse=True):
        if len(segment.get("points", [])) < 2:
            continue
        line = LineString(segment["points"])
        if line.length <= 1e-9:
            continue
        if boundary_union is not None and line.length > 0.0 and line.intersection(boundary_union.buffer(0.05 * upm)).length >= 0.80 * line.length:
            stats["segments_dropped"] += 1
            continue
        if any(_parallel_duplicate(line, LineString(other["points"]), 0.10 * upm) for other in kept_segments):
            stats["segments_dropped"] += 1
            continue
        kept_segments.append(segment)

    fixed_boxes = [_text_box(item) for item in obstacle_texts if item.get("type") == "text" and item.get("point") is not None]
    placed_boxes: list[Polygon] = []
    final_texts: list[dict[str, object]] = []
    order = sorted((index for index in range(len(merged_shapes)) if labels[index] is not None), key=lambda index: merged_shapes[index].area, reverse=True)
    for index in order:
        text = dict(labels[index])
        polygon = merged_shapes[index]
        box = _text_box(text)
        if any(box.intersects(other) for other in fixed_boxes + placed_boxes):
            height = float(text.get("height", 0.1 * upm))
            step = max(height * 0.5, 0.02 * upm)
            minx, miny, maxx, maxy = polygon.bounds
            candidates: list[tuple[float, float, bool]] = []
            centre = polygon.representative_point()
            for x in np.arange(minx, maxx + 1e-9, step):
                for y in np.arange(miny, maxy + 1e-9, step):
                    if not polygon.contains(Point(x, y)):
                        continue
                    trial = _text_box({**text, "point": [float(x), float(y)]})
                    if any(trial.intersects(other) for other in fixed_boxes + placed_boxes):
                        continue
                    candidates.append((float(x), float(y), polygon.buffer(0.02 * upm).contains(trial)))
            if candidates:
                inside = [item for item in candidates if item[2]] or candidates
                x, y, _flag = min(inside, key=lambda item: math.hypot(item[0] - centre.x, item[1] - centre.y))
                text["point"] = [x, y]
                stats["labels_moved"] += 1
            elif polygon.area < min_label_area_m2 * upm**2:
                stats["texts_dropped"] += 1
                continue
        placed_boxes.append(_text_box(text))
        final_texts.append(text)
    result = merged_polygons + kept_segments + final_texts + others
    stats["after"] = len(result)
    return result, stats


def _text_box(text: dict[str, object]) -> Polygon:
    x, y = text["point"]
    height = float(text.get("height", 0.1))
    half_w = 0.5 * len(str(text.get("text", ""))) * 0.62 * height + 0.1 * height
    half_h = 0.5 * height * 1.1
    return Polygon(((x - half_w, y - half_h), (x + half_w, y - half_h), (x + half_w, y + half_h), (x - half_w, y + half_h)))


def _rectangle_or_ell(polygon: Polygon, units_per_metre: float) -> Polygon:
    """Return a rectangle, else an L-shape / simple polygon of at most six corners."""
    rectangle = polygon.minimum_rotated_rectangle
    if isinstance(rectangle, Polygon) and rectangle.area > 0.0:
        inter = rectangle.intersection(polygon).area
        if inter >= 0.80 * polygon.area and rectangle.area - inter <= 0.25 * polygon.area:
            return rectangle
    for tolerance in (0.03, 0.05, 0.08):
        simple = polygon.simplify(tolerance * units_per_metre, preserve_topology=True)
        if isinstance(simple, Polygon) and simple.is_valid and not simple.is_empty and len(simple.exterior.coords) - 1 <= 6:
            return simple
    simple = polygon.simplify(0.03 * units_per_metre, preserve_topology=True)
    return simple if isinstance(simple, Polygon) and simple.is_valid and not simple.is_empty else polygon


def _parallel_duplicate(line: LineString, other: LineString, tolerance: float) -> bool:
    """True when ``line`` (the shorter one) runs parallel to ``other`` within ``tolerance`` along most of its length."""
    if line.length <= 1e-9 or other.length <= 1e-9:
        return False
    a, b = np.asarray(line.coords[0]), np.asarray(line.coords[-1])
    c, d = np.asarray(other.coords[0]), np.asarray(other.coords[-1])
    first, second = (b - a) / line.length, (d - c) / other.length
    if abs(float(first[0] * second[1] - first[1] * second[0])) > math.sin(math.radians(6.0)):
        return False
    samples = [line.interpolate(fraction, normalized=True) for fraction in np.linspace(0.0, 1.0, 9)]
    return float(np.mean([other.distance(point) <= tolerance for point in samples])) >= 0.70


def _or_neighbourhood(bits: np.ndarray) -> np.ndarray:
    padded = np.pad(bits, 1, mode="constant")
    result = np.zeros_like(bits, dtype=np.uint32)
    for row in range(3):
        for column in range(3):
            result |= padded[row:row + bits.shape[0], column:column + bits.shape[1]]
    return result


def _component_lines(component: np.ndarray, grid: RasterGrid, theta: float, units_per_metre: float, settings: VerticalSettings) -> list[LineString]:
    image = (component.astype(np.uint8) * 255)
    minimum = max(2, int(round(settings.min_length_m * units_per_metre / grid.pixel_size)))
    raw = cv2.HoughLinesP(image, 1, np.pi / 180.0, threshold=max(4, minimum // 2), minLineLength=minimum, maxLineGap=max(1, int(round(settings.max_gap_m * units_per_metre / grid.pixel_size))))
    candidates: list[LineString] = []
    if raw is not None:
        for x1, y1, x2, y2 in raw[:, 0]:
            candidates.append(LineString((grid.pixel_to_local(float(y1), float(x1)), grid.pixel_to_local(float(y2), float(x2)))))
    if not candidates:
        rows, columns = np.nonzero(component)
        if len(rows) >= 2:
            points = np.asarray([grid.pixel_to_local(float(row), float(column)) for row, column in zip(rows, columns, strict=True)])
            direction = _principal_axis(points)
            projected = points @ direction
            candidates.append(LineString((tuple(direction * projected.min() + points.mean(axis=0) - direction * (points.mean(axis=0) @ direction)), tuple(direction * projected.max() + points.mean(axis=0) - direction * (points.mean(axis=0) @ direction)))))
    valid: list[LineString] = []
    for line in candidates:
        direction = np.asarray(line.coords[-1]) - np.asarray(line.coords[0])
        angle = math.atan2(float(direction[1]), float(direction[0])) % math.pi
        target = min((theta % math.pi, (theta + math.pi / 2.0) % math.pi), key=lambda value: _angle_distance(angle, value))
        if _angle_distance(angle, target) <= math.radians(settings.axis_tolerance_deg):
            valid.append(_snap_line(line, target))
    return valid


def _line_rejection(line: LineString, component: np.ndarray, grid: RasterGrid, walls: list[LineString], furniture: Iterable[Polygon], countertop: np.ndarray, accepted: list[LineString], units_per_metre: float, settings: VerticalSettings) -> str | None:
    if line.length < settings.min_length_m * units_per_metre:
        return "short"
    samples = _line_samples(line, grid.pixel_size / 2.0)
    support = np.asarray([_sample_mask(component, grid, point, 1.5 * grid.pixel_size) for point in samples])
    if float(support.mean()) < settings.support_fraction:
        return "support"
    points = np.asarray([grid.pixel_to_local(float(row), float(column)) for row, column in zip(*np.nonzero(component), strict=True)])
    nearby = np.asarray([line.distance(Point(point)) <= 2.0 * grid.pixel_size for point in points])
    if nearby.any() and _rms_to_line(points[nearby], line) > settings.rms_m * units_per_metre:
        return "rms"
    if any(line.intersection(poly.boundary.buffer(settings.furniture_distance_m * units_per_metre)).length >= 0.50 * line.length for poly in furniture):
        return "furniture_edge"
    attached = any(line.distance(wall) <= settings.isolation_distance_m * units_per_metre for wall in walls) or any(line.distance(other) <= settings.isolation_distance_m * units_per_metre for other in accepted)
    attached |= any(_sample_mask(countertop, grid, point, 0.10 * units_per_metre) for point in samples)
    return None if attached else "isolated"


def _counter_polygon(line: LineString, countertop: np.ndarray, grid: RasterGrid, theta: float, walls: list[LineString], units_per_metre: float, settings: VerticalSettings) -> Polygon | None:
    if not any(_sample_mask(countertop, grid, point, 0.12 * units_per_metre) for point in _line_samples(line, grid.pixel_size)):
        return None
    near_wall = min((line.distance(wall) for wall in walls), default=float("inf"))
    if not 0.35 * units_per_metre <= near_wall <= 0.75 * units_per_metre:
        return None
    # A buffered front produces a regular axis-aligned closed counter run.  The
    # depth is constrained by the measured wall distance above.
    depth = min(max(near_wall, 0.35 * units_per_metre), 0.75 * units_per_metre)
    polygon = line.buffer(depth, cap_style=2, join_style=2)
    return polygon if isinstance(polygon, Polygon) and polygon.area > 0.0 else None


def _deduplicate(entities: list[dict[str, object]], polygons: list[Polygon], tolerance: float) -> tuple[list[dict[str, object]], list[Polygon]]:
    kept_polygons: list[Polygon] = []
    kept_entities: list[dict[str, object]] = []
    for entity in entities:
        if entity.get("_einb_kind") != "counter":
            kept_entities.append(entity)
            continue
        polygon = Polygon(entity["points"])
        if any(polygon.distance(other) <= tolerance and polygon.intersection(other).area >= 0.5 * min(polygon.area, other.area) for other in kept_polygons):
            continue
        kept_polygons.append(polygon)
        kept_entities.append(entity)
    return kept_entities, kept_polygons or polygons


def _principal_axis(points: np.ndarray) -> np.ndarray:
    centred = points - points.mean(axis=0)
    vector = np.linalg.eigh(centred.T @ centred)[1][:, -1]
    return vector / max(float(np.linalg.norm(vector)), 1e-12)


def _snap_line(line: LineString, angle: float) -> LineString:
    direction = np.asarray((math.cos(angle), math.sin(angle)))
    points = np.asarray(line.coords)
    centre = points.mean(axis=0)
    projected = (points - centre) @ direction
    return LineString((tuple(centre + projected.min() * direction), tuple(centre + projected.max() * direction)))


def _angle_distance(first: float, second: float) -> float:
    return abs((first - second + math.pi / 2.0) % math.pi - math.pi / 2.0)


def _line_samples(line: LineString, spacing: float) -> list[Point]:
    return [line.interpolate(distance) for distance in np.linspace(0.0, line.length, max(2, int(math.ceil(line.length / max(spacing, 1e-9))) + 1))]


def _rms_to_line(points: np.ndarray, line: LineString) -> float:
    return float(math.sqrt(np.mean([line.distance(Point(point)) ** 2 for point in points])))


def _sample_mask(mask: np.ndarray, grid: RasterGrid, point: Point, radius: float) -> bool:
    row, column = grid.local_to_pixel(float(point.x), float(point.y))
    pixels = max(1, int(math.ceil(radius / grid.pixel_size)))
    r0, r1 = max(0, int(row) - pixels), min(mask.shape[0], int(row) + pixels + 1)
    c0, c1 = max(0, int(column) - pixels), min(mask.shape[1], int(column) + pixels + 1)
    return bool(mask[r0:r1, c0:c1].any())


def _line_mask(shape: tuple[int, int], grid: RasterGrid, lines: Iterable[LineString], width: float) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.uint8)
    for line in lines:
        points = np.asarray([grid.local_to_pixel(float(x), float(y))[::-1] for x, y in line.coords], dtype=np.int32)
        if len(points) >= 2:
            cv2.polylines(mask, [points], False, 1, thickness=max(1, int(round(2.0 * width / grid.pixel_size))))
    return mask.astype(bool)


def _polygon_pixels(polygon: Polygon, grid: RasterGrid) -> np.ndarray:
    mask = _mask_polygon(np.zeros((grid.height, grid.width), dtype=bool), grid, polygon=polygon)
    return mask


def _mask_polygon(mask: np.ndarray, grid: RasterGrid, polygon: Polygon | None = None) -> Polygon | np.ndarray:
    if polygon is None:
        polygons = mask_to_polygons(mask, grid)
        return unary_union(polygons) if polygons else Polygon()
    output = np.zeros(mask.shape, dtype=np.uint8)
    points = np.asarray([grid.local_to_pixel(float(x), float(y))[::-1] for x, y in polygon.exterior.coords], dtype=np.int32)
    cv2.fillPoly(output, [points], 1)
    return output.astype(bool)


def _points(polygon: Polygon) -> list[list[float]]:
    return [[float(x), float(y)] for x, y in polygon.exterior.coords[:-1]]
