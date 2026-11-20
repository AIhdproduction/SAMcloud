"""Keep-clear zones of doors and the kitchen-only rule for built-ins (A-EINB / A-OBEN).

Door frames and open leaves are door-class points; the built-in detectors (niches, upper cabinets, built-in fronts) can mistake them for
cabinets.  A door is an opening, so nothing is built in its span, its frame or the swept open leaf.  Upper cabinets and niches belong to
the kitchen cluster (the room that holds the cooktop / sink) only, never to a bath or a corridor.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
from shapely.geometry import LineString, Polygon


@dataclass(frozen=True, slots=True)
class DoorZoneSettings:
    margin_m: float = 0.10  # the span of a door / passage is widened by this much on every side (frame, casing)
    leaf_buffer_m: float = 0.12  # a measured open leaf is a band of this half-width
    max_overlap_share: float = 0.25  # a built-in covered this much by a door zone is a door part, not a cabinet
    kitchen_zone_margin_m: float = 0.80  # the kitchen zone is the room outline, which stops at the counter front: units above the run lie behind it


def door_keep_clear_zones(openings: list[Any], leaves: list[LineString], units_per_metre: float, settings: DoorZoneSettings) -> list[Polygon]:
    """Span (width x wall thickness + margin) of every door / passage and the band of every measured open leaf, in plan units."""
    zones: list[Polygon] = []
    margin = settings.margin_m * units_per_metre
    for opening in openings:
        if opening.kind not in {"DOOR", "passage"}:
            continue
        axis = np.asarray((math.cos(opening.axis_angle), math.sin(opening.axis_angle)))
        normal = np.asarray((-axis[1], axis[0]))
        centre = np.asarray(opening.center, dtype=float)
        half_width, half_depth = opening.width / 2.0 + margin, opening.wall_thickness / 2.0 + margin
        corners = [centre - axis * half_width - normal * half_depth, centre + axis * half_width - normal * half_depth, centre + axis * half_width + normal * half_depth, centre - axis * half_width + normal * half_depth]
        zones.append(Polygon([tuple(float(v) for v in corner) for corner in corners]))
    zones.extend(leaf.buffer(settings.leaf_buffer_m * units_per_metre) for leaf in leaves)
    return zones


def drop_misplaced_builtins(
    layers: dict[str, list[dict[str, Any]]], openings: list[Any], leaves: list[LineString], kitchen_zone: Any | None, units_per_metre: float,
    settings: DoorZoneSettings = DoorZoneSettings(),
) -> dict[str, int]:
    """Remove built-in polygons that stand on a door (span / frame / open leaf) and niches / upper cabinets outside the kitchen.

    Kitchen runs and islands are exempt from the door test (a counter run may end at a door casing).  ``kitchen_zone`` None (no kitchen
    evidence at all) disables the outside-the-kitchen test.  Returns the number of dropped entities per reason.
    """
    zones = door_keep_clear_zones(openings, leaves, units_per_metre, settings)
    if kitchen_zone is not None:
        kitchen_zone = kitchen_zone.buffer(settings.kitchen_zone_margin_m * units_per_metre)
    dropped = {"builtins_dropped_at_door": 0, "builtins_dropped_outside_kitchen": 0}
    kitchen_only = {"niche", "upper_cabinet"}
    for name in ("A-EINB", "A-OBEN"):
        kept: list[dict[str, Any]] = []
        for entity in layers.get(name, []):
            if entity.get("type") != "polyline" or not entity.get("closed") or len(entity.get("points", [])) < 3:
                kept.append(entity)
                continue
            kind = entity.get("_einb_kind") or ("upper_cabinet" if entity.get("_upper_cabinet") else None)
            polygon = Polygon(entity["points"])
            if not polygon.is_valid or polygon.area <= 0.0:
                kept.append(entity)
                continue
            if kind not in {"island", "fine_kitchen"} and any(polygon.intersection(zone).area >= settings.max_overlap_share * polygon.area for zone in zones):
                dropped["builtins_dropped_at_door"] += 1
                continue
            if kind in kitchen_only and kitchen_zone is not None and polygon.intersection(kitchen_zone).area < 0.5 * polygon.area:
                dropped["builtins_dropped_outside_kitchen"] += 1
                continue
            kept.append(entity)
        layers[name] = kept
    return dropped
