"""Semantic room names (Bad, Küche, Schlafen, Wohnen ...) from the fixtures inside a room."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

from shapely.geometry import Point, Polygon

SANITARY_TAGS = frozenset({"toilet", "washbasin", "bathtub", "shower"})

_WORD_TAGS: dict[str, str] = {
    "wc": "toilet", "toilet": "toilet", "wc_wall": "toilet",
    "washbasin": "washbasin", "lavabo": "washbasin", "waschbecken": "washbasin",
    "bathtub": "bathtub", "badewanne": "bathtub",
    "shower": "shower", "shower tray": "shower", "dusche": "shower",
    "kitchen counter": "kitchen", "kitchen cabinet": "kitchen", "cooktop": "kitchen", "stove": "kitchen", "sink": "kitchen",
    "kitchen sink": "kitchen", "küche": "kitchen", "herd": "kitchen", "spüle": "kitchen",
    "bed": "bed", "bett": "bed", "mattress": "bed",
    "wardrobe": "wardrobe", "schrank": "wardrobe", "kleiderschrank": "wardrobe",
    "sofa": "living", "couch": "living", "tv": "living", "armchair": "living", "sessel": "living",
    "dining table": "dining", "esstisch": "dining", "table": "dining", "tisch": "dining",
    "desk": "office", "schreibtisch": "office", "monitor": "office", "computer": "office", "bildschirm": "office",
}
_BLOCK_PREFIX_TAGS = (
    ("WC_WALL", "toilet"), ("WASHBASIN", "washbasin"), ("BATHTUB", "bathtub"), ("SHOWER", "shower"),
    ("COOKTOP", "kitchen"), ("KITCHEN_", "kitchen"), ("BED_", "bed"), ("WARDROBE", "wardrobe"), ("SOFA", "living"), ("TABLE_DINING", "dining"),
)


@dataclass(frozen=True, slots=True)
class RoomNameSettings:
    """Room labelling thresholds (metric)."""

    min_room_area_m2: float = 2.0
    guest_wc_max_area_m2: float = 4.0
    hall_max_area_m2: float = 15.0
    hall_min_doors: int = 3
    min_hall_area_m2: float = 1.0  # a smaller room than min_room_area_m2 is kept when it is a hall (corridor between doors)
    office_min_items: int = 2  # desks / monitors needed before a room is named "Büro"


@dataclass(frozen=True, slots=True)
class RoomName:
    """Name decision for one room polygon."""

    name: str | None
    area_m2: float
    keep: bool
    tags: frozenset[str]


def tag_for_name(name: str) -> str | None:
    """Map a block name, German display name or English fine label to a semantic tag."""
    text = str(name or "").strip()
    upper = text.upper()
    for prefix, tag in _BLOCK_PREFIX_TAGS:
        if upper.startswith(prefix):
            return tag
    key = text.lower().split(" h=")[0].strip()
    return _WORD_TAGS.get(key)


def classify_room(tags: Iterable[str], area_m2: float, settings: RoomNameSettings = RoomNameSettings()) -> str | None:
    """Return the semantic room name for a set of fixture tags, or None without evidence."""
    found = set(tags)
    if found & SANITARY_TAGS:
        if found & {"bathtub", "shower"}:
            return "Bad"
        if "toilet" in found and area_m2 < settings.guest_wc_max_area_m2:
            return "Gäste-WC"
        return "Bad"
    if "kitchen" in found:
        if "living" in found:
            return "Wohnen / Küche"
        return "Küche / Essen" if "dining" in found else "Küche"
    if "bed" in found:
        return "Schlafen"
    if "office" in found:
        return "Büro"
    if "wardrobe" in found:
        return "Zimmer"
    if "living" in found:
        return "Wohnen"
    if "dining" in found:
        return "Essen"
    return None


def room_label_text(name: str, area_m2: float) -> str:
    """Room text format, e.g. ``Küche  12.4 m²`` (two spaces before the area)."""
    return f"{name}  {area_m2:.1f} m²"


def name_rooms(
    rooms: list[Polygon],
    tagged_points: Iterable[tuple[tuple[float, float], str]],
    units_per_metre: float,
    settings: RoomNameSettings = RoomNameSettings(),
) -> list[RoomName]:
    """Name every room from the tagged fixture points inside it.

    A room below ``min_room_area_m2`` without sanitary evidence is not a real
    room (for example the inside of a wardrobe niche): ``keep`` is False.
    """
    upm = max(float(units_per_metre), 1e-9)
    points = [(Point(float(x), float(y)), tag) for (x, y), tag in tagged_points]
    result: list[RoomName] = []
    for polygon in rooms:
        area = polygon.area / (upm * upm)
        zone = polygon.buffer(0.05 * upm)
        inside = [tag for point, tag in points if zone.contains(point)]
        tags = frozenset(tag for tag in inside if tag != "office" or inside.count("office") >= settings.office_min_items)
        keep = area >= settings.min_room_area_m2 or bool(tags & SANITARY_TAGS)
        result.append(RoomName(classify_room(tags, area, settings), area, keep, tags))
    return result


def name_halls(
    rooms: list[Polygon],
    decisions: list[RoomName],
    doors: Iterable[tuple[tuple[float, float], float]],
    units_per_metre: float,
    settings: RoomNameSettings = RoomNameSettings(),
) -> list[RoomName]:
    """Name an unnamed circulation room "Flur" (hall / corridor).

    A room without fixture evidence is a hall when it holds the entrance door (a
    door with a room on one side and no room on the other) and at least one more
    door, or when at least ``hall_min_doors`` doors open into it; it must not be
    larger than ``hall_max_area_m2``.  A room named only by weak evidence ("Zimmer" from a wardrobe,
    "Büro") is still a hall, and a corridor below ``min_room_area_m2`` is kept as "Flur" from
    ``min_hall_area_m2``.  ``doors`` are (centre, axis angle) pairs.
    """
    upm = max(float(units_per_metre), 1e-9)
    door_list = [((float(centre[0]), float(centre[1])), float(angle)) for centre, angle in doors]
    reach = 0.45 * upm
    probe = 0.60 * upm

    def in_room(x: float, y: float) -> bool:
        point = Point(x, y)
        return any(room.covers(point) for room in rooms)

    entrance: list[bool] = []
    for (x, y), angle in door_list:
        nx, ny = -math.sin(angle), math.cos(angle)
        sides = [in_room(x + sign * nx * probe, y + sign * ny * probe) for sign in (1.0, -1.0)]
        entrance.append(sides[0] != sides[1])
    output: list[RoomName] = []
    for polygon, decision in zip(rooms, decisions, strict=True):
        small = not decision.keep and decision.area_m2 >= settings.min_hall_area_m2
        if decision.name not in (None, "Zimmer", "Büro") or not (decision.keep or small) or decision.area_m2 > settings.hall_max_area_m2 or decision.tags - {"wardrobe", "office"}:
            output.append(decision)
            continue
        touching = [index for index, ((x, y), _angle) in enumerate(door_list) if polygon.distance(Point(x, y)) <= reach]
        has_entrance = any(entrance[index] for index in touching)
        if (has_entrance and len(touching) >= 2) or len(touching) >= settings.hall_min_doors:
            output.append(RoomName("Flur", decision.area_m2, True, decision.tags))
        else:
            output.append(decision)
    return output
