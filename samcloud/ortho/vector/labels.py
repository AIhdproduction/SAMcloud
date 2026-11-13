"""Smart text label placement inside polygons with axis alignment, padding, and overlap prevention."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
import math
from pathlib import Path

import numpy as np
from shapely import affinity
from shapely.errors import GEOSException
from shapely.geometry import Polygon, box
from shapely.ops import polylabel
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
