"""Smart text label placement inside polygons with axis alignment, padding, and overlap prevention."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
import math
from pathlib import Path

import numpy as np
from shapely import affinity
from shapely.errors import GEOSException
from shapely.geometry import Point, Polygon, box
from shapely.ops import polylabel, unary_union
import yaml


@dataclass(frozen=True, slots=True)
class LabelSettings:
    """Settings controlling label size scaling, placement, padding, and shrinking."""

    base_height_factor: float = 0.12
    min_height: float = 0.08
    max_height: float = 0.60
    min_area: float = 1.0
    padding_factor: float = 0.15
    char_width_factor: float = 0.62
    shrink_step: float = 0.85

    def __post_init__(self) -> None:
        if self.base_height_factor <= 0.0:
            raise ValueError("base_height_factor must be positive")
        if self.min_height <= 0.0:
            raise ValueError("min_height must be positive")
        if self.max_height < self.min_height:
            raise ValueError("max_height cannot be smaller than min_height")
        if self.min_area < 0.0:
            raise ValueError("min_area cannot be negative")
        if self.padding_factor < 0.0:
            raise ValueError("padding_factor cannot be negative")
        if self.char_width_factor <= 0.0:
            raise ValueError("char_width_factor must be positive")
        if not (0.0 < self.shrink_step < 1.0):
            raise ValueError("shrink_step must be between 0 and 1")


def class_display_name(raw_name: str) -> str:
    """Convert raw prompt lists or class names into short, capitalized display labels."""
    if not raw_name or not raw_name.strip():
        return ""
    first = raw_name.split(",")[0].strip()
    if not first:
        return ""
    return first[0].upper() + first[1:]


def _label_box(entity: dict, settings: LabelSettings) -> Polygon:
    """Compute the padded bounding box of a placed label entity."""
    cx, cy = entity["point"]
    h = float(entity["height"])
    rot = float(entity["rotation"])
    text = entity["text"]
    w = len(text) * settings.char_width_factor * h
    pad = settings.padding_factor * h
    hw = w / 2.0 + pad
    hh = h / 2.0 + pad
    raw = box(cx - hw, cy - hh, cx + hw, cy + hh)
    return affinity.rotate(raw, rot, origin=(cx, cy))


def place_label(polygon: Polygon, text: str, settings: LabelSettings) -> dict | None:
    """Place a single text label inside a polygon following its dominant axis."""
    if not isinstance(polygon, Polygon) or polygon.is_empty:
        return None
    if polygon.area < settings.min_area:
        return None

    display_text = text.strip()
    if not display_text:
        return None

    min_rect = polygon.minimum_rotated_rectangle
    if min_rect.is_empty:
        return None

    coords = list(min_rect.exterior.coords)
    if len(coords) < 4:
        return None

    corners = [np.asarray(c, dtype=np.float64) for c in coords[:4]]
    e0 = corners[1] - corners[0]
    e1 = corners[2] - corners[1]
    len0 = float(np.hypot(e0[0], e0[1]))
    len1 = float(np.hypot(e1[0], e1[1]))

    if len0 >= len1:
        long_edge = e0
        long_side = len0
        short_side = len1
    else:
        long_edge = e1
        long_side = len1
        short_side = len0

    if long_side <= 1e-9 or short_side <= 1e-9:
        return None

    angle_rad = math.atan2(float(long_edge[1]), float(long_edge[0]))
    angle_deg = math.degrees(angle_rad)

    while angle_deg > 90.0:
        angle_deg -= 180.0
    while angle_deg <= -90.0:
        angle_deg += 180.0

    start_height = min(settings.max_height, max(settings.min_height, settings.base_height_factor * short_side))
    h = start_height

    try:
        pole = polylabel(polygon, tolerance=0.01)
        anchor_poly = (float(pole.x), float(pole.y))
    except GEOSException:
        centroid = polygon.centroid
        anchor_poly = (float(centroid.x), float(centroid.y))

    rect_center = min_rect.centroid
    anchor_rect = (float(rect_center.x), float(rect_center.y))

    rot_perp = angle_deg + 90.0
    while rot_perp > 90.0:
        rot_perp -= 180.0
    while rot_perp <= -90.0:
        rot_perp += 180.0

    def _test_box(center: tuple[float, float], height: float, rotation: float) -> Polygon:
        w = len(display_text) * settings.char_width_factor * height
        pad = settings.padding_factor * height
        hw = w / 2.0 + pad
        hh = height / 2.0 + pad
        cx, cy = center
        raw_box = box(cx - hw, cy - hh, cx + hw, cy + hh)
        return affinity.rotate(raw_box, rotation, origin=(cx, cy))

    while h >= settings.min_height - 1e-9:
        text_w_padded = len(display_text) * settings.char_width_factor * h + 2.0 * settings.padding_factor * h
        allow_perp = (text_w_padded <= short_side + 1e-6)

        candidates = [
            (anchor_poly, angle_deg),
            (anchor_rect, angle_deg),
        ]
        if allow_perp:
            candidates.append((anchor_poly, rot_perp))
            candidates.append((anchor_rect, rot_perp))

        for center, rot in candidates:
            candidate_box = _test_box(center, h, rot)
            if polygon.covers(candidate_box):
                return {
                    "type": "text",
                    "text": display_text,
                    "point": [float(center[0]), float(center[1])],
                    "height": float(h),
                    "rotation": float(rot),
                    "align": "MIDDLE_CENTER",
                    "layer_role": "label",
                }

        h *= settings.shrink_step

    return None


def label_entities(
    items: Iterable[tuple[Polygon, str]],
    settings: LabelSettings,
) -> list[dict]:
    """Place labels for multiple polygons, prioritizing larger polygons and avoiding overlaps."""
    valid_items = [
        (p, t) for p, t in items
        if isinstance(p, Polygon) and not p.is_empty and p.area >= settings.min_area
    ]
    sorted_items = sorted(valid_items, key=lambda item: item[0].area, reverse=True)

    placed_boxes: list[Polygon] = []
    entities: list[dict] = []

    for polygon, text in sorted_items:
        entity = place_label(polygon, text, settings)
        if entity is None:
            continue

        lbl_box = _label_box(entity, settings)

        overlaps = False
        for existing in placed_boxes:
            if lbl_box.intersects(existing):
                inter = lbl_box.intersection(existing)
                if inter.area > 1e-6:
                    overlaps = True
                    break

        if overlaps:
            continue

        placed_boxes.append(lbl_box)
        entities.append(entity)

    return entities


_DIRECTIONS: tuple[tuple[float, float], ...] = tuple(
    (math.cos(math.radians(angle)), math.sin(math.radians(angle))) for angle in range(0, 360, 45)
)
_FIXED_LABEL_LAYERS = frozenset({"A-DOOR"})
_MOVABLE_LABEL_LAYERS = frozenset({"A-ANNO", "A-EINB"})


def _text_box(entity: dict, height: float, point: tuple[float, float], settings: LabelSettings) -> Polygon:
    """Unpadded rectangle of a text entity: len(text) * char_width_factor * height wide, height tall, rotated about its centre."""
    width = len(str(entity.get("text", ""))) * settings.char_width_factor * height
    cx, cy = point
    raw = box(cx - width / 2.0, cy - height / 2.0, cx + width / 2.0, cy + height / 2.0)
    rotation = float(entity.get("rotation", 0.0))
    return affinity.rotate(raw, rotation, origin=(cx, cy)) if abs(rotation) > 1e-12 else raw


def _overlaps(candidate: Polygon, others: Iterable[Polygon], minimum_area: float) -> bool:
    c = candidate.bounds
    for other in others:
        o = other.bounds
        if c[0] > o[2] or o[0] > c[2] or c[1] > o[3] or o[1] > c[3]:
            continue
        if candidate.intersection(other).area > minimum_area:
            return True
    return False


def _room_containing(point: tuple[float, float], rooms: list[Polygon]) -> Polygon | None:
    location = Point(point)
    containing = [room for room in rooms if room.covers(location)]
    return min(containing, key=lambda room: room.area) if containing else None


def _free_room_point(
    entity: dict, room: Polygon, obstacles: list[Polygon], blockers: list[Polygon], upm: float, minimum_area: float,
    settings: LabelSettings,
) -> tuple[float, float] | None:
    """Best point inside the room where the label box touches neither an obstacle nor a blocking box."""
    height = float(entity["height"])
    free = room.difference(unary_union([obstacle.buffer(0.02 * upm) for obstacle in obstacles])) if obstacles else room
    pieces = [piece for piece in getattr(free, "geoms", [free]) if isinstance(piece, Polygon) and not piece.is_empty]
    original = (float(entity["point"][0]), float(entity["point"][1]))
    best: tuple[float, float, tuple[float, float]] | None = None
    for piece in pieces:
        try:
            pole = polylabel(piece, tolerance=0.01 * upm)
        except GEOSException:
            continue
        point = (float(pole.x), float(pole.y))
        candidate = _text_box(entity, height, point, settings)
        if not room.buffer(1e-9).covers(candidate) or _overlaps(candidate, obstacles, minimum_area) or _overlaps(candidate, blockers, minimum_area):
            continue
        clearance = float(pole.distance(piece.exterior))
        key = (-clearance, math.hypot(point[0] - original[0], point[1] - original[1]))
        if best is None or key < best[:2]:
            best = (key[0], key[1], point)
    if best is None:
        # The pole of a long free strip can sit where the label box does not fit: scan the room.
        step = max(0.5 * height, 0.05 * upm)
        min_x, min_y, max_x, max_y = room.bounds
        free_union = unary_union(pieces) if pieces else None
        for x in np.arange(min_x + step / 2.0, max_x, step):
            for y in np.arange(min_y + step / 2.0, max_y, step):
                point = (float(x), float(y))
                candidate = _text_box(entity, height, point, settings)
                if not room.buffer(1e-9).covers(candidate) or _overlaps(candidate, obstacles, minimum_area) or _overlaps(candidate, blockers, minimum_area):
                    continue
                clearance = float(free_union.exterior.distance(Point(point))) if isinstance(free_union, Polygon) else 0.0
                key = (-clearance, math.hypot(point[0] - original[0], point[1] - original[1]))
                if best is None or key < best[:2]:
                    best = (key[0], key[1], point)
    return best[2] if best is not None else None


def resolve_label_overlaps(
    layers: dict[str, list[dict]],
    obstacles: list[Polygon],
    rooms: list[Polygon],
    units_per_metre: float,
    settings: LabelSettings,
) -> dict[str, int]:
    """Resolve overlaps between plan text entities in place; never deletes a label.

    A-DOOR texts and texts on other layers stay where they are. Room names (A-ANNO with ``_room_name``) stay unless they
    cover an obstacle (insert footprint) or cross their room outline; then they move to the best free point inside their room. Other A-ANNO and A-EINB
    labels are movable: on overlap (or when their box leaves their room) they take the nearest free offset (8 directions, steps of 1.2 text heights, at most
    0.6 m), else shrink by ``settings.shrink_step`` and retry, else keep their position.
    """
    upm = max(float(units_per_metre), 1e-9)
    minimum_area = 1e-6 * upm * upm
    max_shift = 0.6 * upm
    height_floor = max(0.06 * upm, settings.min_height * upm)
    metrics = {"labels_moved": 0, "labels_shrunk": 0, "labels_unresolved": 0}

    fixed: list[dict] = []
    room_labels: list[dict] = []
    movable: list[dict] = []
    for name, entities in layers.items():
        for entity in entities:
            if entity.get("type") != "text" or not str(entity.get("text", "")).strip() or "point" not in entity:
                continue
            if float(entity.get("height", 0.0)) <= 0.0:
                continue
            if name in _FIXED_LABEL_LAYERS or name not in _MOVABLE_LABEL_LAYERS:
                fixed.append(entity)
            elif "_room_name" in entity:
                room_labels.append(entity)
            else:
                movable.append(entity)

    def current_box(entity: dict) -> Polygon:
        return _text_box(entity, float(entity["height"]), (float(entity["point"][0]), float(entity["point"][1])), settings)

    blockers = [current_box(entity) for entity in fixed]
    placed: list[Polygon] = list(blockers)

    for entity in sorted(room_labels, key=lambda item: (-float(item["height"]), str(item["text"]))):
        label_box = current_box(entity)
        origin = (float(entity["point"][0]), float(entity["point"][1]))
        own_room = _room_containing(origin, rooms)
        # A room name must lie inside its room: a box across the room outline sits under the wall.
        outside = own_room is not None and not own_room.buffer(1e-9).covers(label_box)
        if (obstacles and _overlaps(label_box, obstacles, minimum_area)) or outside:
            room = own_room
            target = _free_room_point(entity, room, obstacles, blockers, upm, minimum_area, settings) if room is not None else None
            # A small room (WC, bath) may have no free spot for the full-size name: shrink it step by step.
            original_height = float(entity["height"])
            while target is None and room is not None and float(entity["height"]) * settings.shrink_step >= height_floor - 1e-9:
                entity["height"] = float(entity["height"]) * settings.shrink_step
                target = _free_room_point(entity, room, obstacles, blockers, upm, minimum_area, settings)
            if target is not None:
                entity["point"] = [float(target[0]), float(target[1])]
                label_box = current_box(entity)
                metrics["labels_moved"] += 1
                if float(entity["height"]) != original_height:
                    metrics["labels_shrunk"] += 1
            else:
                entity["height"] = original_height
                metrics["labels_unresolved"] += 1
        placed.append(label_box)

    for entity in sorted(movable, key=lambda item: (-float(item["height"]), str(item["text"]), float(item["point"][0]), float(item["point"][1]))):
        origin = (float(entity["point"][0]), float(entity["point"][1]))
        original_height = float(entity["height"])
        label_box = current_box(entity)
        home = _room_containing(origin, rooms)
        inside_home = home is None or home.buffer(1e-9).covers(label_box)
        if inside_home and not _overlaps(label_box, placed, minimum_area):
            placed.append(label_box)
            continue
        chosen: tuple[tuple[float, float], float] | None = None
        height = original_height
        first_pass = True
        while chosen is None:
            offsets: list[tuple[float, float]] = [] if first_pass else [(0.0, 0.0)]
            step = 1.2 * height
            for k in range(1, 5):
                if k * step > max_shift + 1e-9:
                    break
                offsets.extend((dx * k * step, dy * k * step) for dx, dy in _DIRECTIONS)
            for dx, dy in offsets:
                point = (origin[0] + dx, origin[1] + dy)
                candidate = _text_box(entity, height, point, settings)
                if home is not None and not home.buffer(1e-9).covers(candidate):
                    continue
                if not _overlaps(candidate, placed, minimum_area):
                    chosen = (point, height)
                    break
            if chosen is not None:
                break
            first_pass = False
            smaller = height * settings.shrink_step
            if smaller < height_floor - 1e-9:
                break
            height = smaller
        if chosen is None:
            metrics["labels_unresolved"] += 1
            placed.append(label_box)
            continue
        point, height = chosen
        if point != origin:
            entity["point"] = [float(point[0]), float(point[1])]
            metrics["labels_moved"] += 1
        if height != original_height:
            entity["height"] = float(height)
            metrics["labels_shrunk"] += 1
        placed.append(current_box(entity))
    return metrics


def _default_config_path() -> Path:
    direct = Path(__file__).resolve().parents[3] / "config" / "ortho_plan.yaml"
    if direct.exists():
        return direct
    cwd_candidate = Path("config/ortho_plan.yaml")
    return cwd_candidate if cwd_candidate.exists() else direct


def load_label_settings(path: Path | str | None = None) -> LabelSettings:
    """Load and validate label settings from ortho_plan.yaml or a custom path."""
    target_path = Path(path) if path is not None else _default_config_path()
    if not target_path.exists():
        return LabelSettings()

    with open(target_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict) or "labels" not in data:
        return LabelSettings()

    section = data["labels"]
    if not isinstance(section, dict):
        return LabelSettings()

    return LabelSettings(
        base_height_factor=float(section.get("base_height_factor", 0.12)),
        min_height=float(section.get("min_height", 0.08)),
        max_height=float(section.get("max_height", 0.60)),
        min_area=float(section.get("min_area", 1.0)),
        padding_factor=float(section.get("padding_factor", 0.15)),
        char_width_factor=float(section.get("char_width_factor", 0.62)),
        shrink_step=float(section.get("shrink_step", 0.85)),
    )
