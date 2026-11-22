"""Furniture the plan lost: scan instances that no plan item represents become plan items.

The 2D plan forms its furniture from the orthophoto; a chair standing close to a table, or a hob inside a kitchen run,
can end up without an item of its own.  The classified cloud still knows them as separate instances (instance id +
label).  An instance is added only with strong evidence and only for classes whose shape the reconstruction handles:

* standing furniture (chairs, armchairs, tables): enough points, standing on the floor, a class-plausible height and
  size, and (almost) none of its points inside another item's outline;
* kitchen inserts (hob, sink): lying on a kitchen base run at worktop height, with no item of the same kind on that run.

All thresholds are metric class priors (furniture dimensions), nothing is project specific.  Added items carry
``fit["added"] = "instance"`` so the 2D DXF can draw them and reviewers can find them.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import shapely
from shapely.geometry import MultiPoint, Polygon


@dataclass(frozen=True, slots=True)
class _Rule:
    label: str  # German display label
    semantic: str  # canonical English label
    catalogue_kind: str | None
    area_m2: tuple[float, float]
    top_m: tuple[float, float]
    min_points: int
    kitchen_insert: bool = False


_RULES: dict[str, _Rule] = {
    "chair": _Rule("Stuhl", "chair", None, (0.08, 0.60), (0.55, 1.25), 600),
    "stool": _Rule("Hocker", "stool", None, (0.06, 0.35), (0.40, 0.90), 500),
    "armchair": _Rule("Sessel", "armchair", None, (0.25, 1.30), (0.50, 1.15), 1200),
    "table": _Rule("Tisch", "table", None, (0.20, 3.50), (0.35, 0.95), 1000),
    "dining table": _Rule("Esstisch", "dining table", None, (0.40, 3.50), (0.60, 0.95), 1000),
    "coffee table": _Rule("Couchtisch", "coffee table", None, (0.15, 1.80), (0.25, 0.60), 800),
    "desk": _Rule("Schreibtisch", "desk", None, (0.40, 2.50), (0.60, 0.95), 1000),
    "stove": _Rule("Herd", "stove", "COOKTOP", (0.10, 1.20), (0.80, 1.10), 300, kitchen_insert=True),
    "kitchen sink": _Rule("Spüle", "kitchen sink", "KITCHEN_SINK", (0.10, 1.20), (0.80, 1.10), 300, kitchen_insert=True),
}
_SAME_KIND = {"COOKTOP": ("cooktop", "herd", "stove", "hob"), "KITCHEN_SINK": ("kitchen_sink", "spüle", "kitchen sink", "sink")}


def load_instance_labels(instances_path: Path | str | None) -> dict[int, str]:
    if instances_path is None or not Path(instances_path).is_file():
        return {}
    try:
        payload = json.loads(Path(instances_path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return {int(item["id"]): str(item.get("label") or "").strip().lower() for item in payload.get("instances", []) if "id" in item}


def _polygon(points: list[list[float]]) -> Polygon | None:
    try:
        polygon = Polygon(points)
        polygon = polygon if polygon.is_valid else polygon.buffer(0)
    except (TypeError, ValueError):
        return None
    return polygon if isinstance(polygon, Polygon) and not polygon.is_empty else None


def _trimmed_rectangle(xy: np.ndarray) -> Polygon | None:
    """Minimum rotated rectangle of the instance's points without stray points (p2..p98 in the rectangle frame)."""
    hull = MultiPoint([tuple(p) for p in np.unique(np.round(xy, 3), axis=0)]).convex_hull
    if not isinstance(hull, Polygon):
        return None
    rectangle = hull.minimum_rotated_rectangle
    corners = np.asarray(rectangle.exterior.coords, dtype=float)[:4]
    u = corners[1] - corners[0]
    if float(np.hypot(*u)) < 1e-9:
        return None
    u = u / np.hypot(*u)
    v = np.asarray((-u[1], u[0]))
    a, b = xy @ u, xy @ v
    a0, a1 = np.percentile(a, [2, 98])
    b0, b1 = np.percentile(b, [2, 98])
    ring = [a0 * u + b0 * v, a1 * u + b0 * v, a1 * u + b1 * v, a0 * u + b1 * v]
    polygon = Polygon(ring)
    return polygon if polygon.is_valid and polygon.area > 0 else None


def discover_missing_items(
    items: list[Any], cloud: Any, instance_labels: dict[int, str], *, units_per_metre: float, floor_z: float, item_factory: Any,
) -> list[Any]:
    """New plan items for well-evidenced scan instances no item represents (appended to ``items``); returns the new items."""
    if cloud is None or not len(cloud.xyz) or not instance_labels:
        return []
    upm = max(float(units_per_metre), 1e-9)
    represented = {int(item.instance_id) for item in items if item.instance_id is not None}
    outlines = [(item, polygon) for item in items if (polygon := _polygon(item.footprint)) is not None]
    kitchen_runs = [polygon for item, polygon in outlines if str(item.catalogue_kind or "").lower() == "kitchen_base"]
    numbers = [int(item.id[1:]) for item in items if str(item.id)[1:].isdigit()]
    next_number = max(numbers or [0]) + 1
    order = np.argsort(cloud.instance, kind="stable")
    sorted_ids = cloud.instance[order]
    added: list[Any] = []
    for identifier, label in sorted(instance_labels.items()):
        rule = _RULES.get(label)
        if rule is None or identifier in represented:
            continue
        start, stop = np.searchsorted(sorted_ids, identifier, side="left"), np.searchsorted(sorted_ids, identifier, side="right")
        if stop - start < rule.min_points:
            continue
        member = cloud.xyz[order[start:stop]]
        z = (member[:, 2] - floor_z) / upm
        z_top = float(np.percentile(z, 98))
        rectangle = _trimmed_rectangle(member[:, :2])
        if rectangle is None:
            continue
        area = rectangle.area / (upm * upm)
        if not rule.area_m2[0] <= area <= rule.area_m2[1] or not rule.top_m[0] <= z_top <= rule.top_m[1]:
            continue
        if rule.kitchen_insert:
            run = max(kitchen_runs, key=lambda polygon: polygon.intersection(rectangle).area, default=None)
            if run is None or rectangle.intersection(run).area < 0.5 * rectangle.area:
                continue
            same = _SAME_KIND.get(rule.catalogue_kind or "", ())
            if any(
                polygon.distance(rectangle) <= 2.5 * upm and any(str(text or "").strip().lower() in same for text in (item.catalogue_kind, item.semantic_label, item.label))
                for item, polygon in outlines
            ):
                continue  # the kitchen already has its hob / sink (a kitchen has one; a second one needs a plan item)
            footprint = rectangle.intersection(run)
            if not isinstance(footprint, Polygon) or footprint.is_empty:
                continue
            z_range = None
        else:
            near_floor = int(np.count_nonzero(z < 0.12))
            if near_floor < max(5, int(0.005 * len(z))):
                continue  # not standing on the floor (a fragment seen only at seat / top height); thin legs give few points
            inside = np.zeros(len(member), dtype=bool)
            for _item, polygon in outlines:
                inside |= shapely.contains_xy(polygon, member[:, 0], member[:, 1])
            if float(inside.mean()) > 0.15:
                continue  # mostly inside an existing item: that item represents it
            footprint = rectangle
            z_range = (floor_z, floor_z + z_top * upm)
        ring = [[float(x), float(y)] for x, y in list(footprint.exterior.coords)[:-1]]
        item = item_factory(
            id=f"F{next_number:03d}", instance_id=int(identifier), label=rule.label, semantic_label=rule.semantic,
            catalogue_kind=rule.catalogue_kind, layer="A-EINB" if rule.kitchen_insert else "A-FURN", footprint=ring,
            z_min=None if z_range is None else z_range[0], z_max=None if z_range is None else z_range[1],
            z_source="default" if z_range is None else "points", default_height_m=None if rule.kitchen_insert else rule.top_m[1],
            default_z_m=[0.86, 0.90] if rule.kitchen_insert else None,
            fit={"added": "instance", "instance_label": label, "instance_points": int(stop - start)},
        )
        next_number += 1
        items.append(item)
        added.append(item)
        outlines.append((item, footprint))
    return added


# canonical English semantic labels of catalogue inserts (the plan stores the German display name there)
_CANONICAL_SEMANTIC = {
    "KITCHEN_SINK": "kitchen sink", "COOKTOP": "stove", "WC_WALL": "toilet", "WC_FLOOR": "toilet", "WC": "toilet",
    "WASHBASIN": "washbasin", "BATHTUB": "bathtub", "SHOWER": "shower tray", "BED_DOUBLE": "bed", "BED_SINGLE": "bed",
}


def canonical_semantic_labels(items: list[Any]) -> int:
    """Catalogue inserts get their canonical English semantic label (the display ``label`` stays German); returns the count."""
    changed = 0
    for item in items:
        canonical = _CANONICAL_SEMANTIC.get(str(item.catalogue_kind or ""))
        if canonical and item.semantic_label != canonical:
            item.semantic_label = canonical
            changed += 1
    return changed


_APPLIANCE_NAMES = frozenset({"refrigerator", "fridge", "kühlschrank", "kuehlschrank", "dishwasher", "washing machine", "oven"})
_ROW_UNITS = frozenset({"tall_unit", "oven_tower", "builtin_front"})


def merge_split_appliances(items: list[Any], cloud: Any, *, units_per_metre: float, cell_m: float = 0.05, min_cover: float = 0.6) -> list[dict[str, Any]]:
    """A kitchen row unit without instance (tall unit) that is really the rest of an appliance the row split cut off.

    When the appliance's own instance points cover most of the neighbouring row unit seen from above, the unit is
    removed and the appliance takes the union of both outlines (its minimum rotated rectangle).  Returns what was merged."""
    if cloud is None or not len(cloud.xyz):
        return []
    upm = max(float(units_per_metre), 1e-9)
    cell = cell_m * upm
    merged: list[dict[str, Any]] = []
    for appliance in [item for item in items if item.instance_id is not None]:
        if not any(str(text or "").strip().lower() in _APPLIANCE_NAMES for text in (appliance.label, appliance.semantic_label, appliance.catalogue_kind)):
            continue
        own = _polygon(appliance.footprint)
        member = cloud.xyz[cloud.instance == int(appliance.instance_id)]
        if own is None or len(member) < 200:
            continue
        occupied = {(int(a), int(b)) for a, b in np.floor(member[:, :2] / cell).astype(np.int64)}
        for unit in [item for item in items if item is not appliance and item.instance_id is None and str(item.catalogue_kind or "") in _ROW_UNITS]:
            polygon = _polygon(unit.footprint)
            if polygon is None or polygon.distance(own) > 0.03 * upm:
                continue
            x0, y0, x1, y1 = polygon.bounds
            xs, ys = np.arange(x0 + cell / 2, x1, cell), np.arange(y0 + cell / 2, y1, cell)
            if not len(xs) or not len(ys):
                continue
            gx, gy = np.meshgrid(xs, ys)
            centres = np.column_stack((gx.ravel(), gy.ravel()))
            centres = centres[shapely.contains_xy(polygon, centres[:, 0], centres[:, 1])]
            if not len(centres):
                continue
            keys = np.floor(centres / cell).astype(np.int64)
            cover = float(np.mean([(int(a), int(b)) in occupied for a, b in keys]))
            if cover < min_cover:
                continue
            union = own.union(polygon).minimum_rotated_rectangle
            if not isinstance(union, Polygon) or union.area > 1.15 * (own.area + polygon.area):
                continue
            appliance.footprint = [[float(x), float(y)] for x, y in list(union.exterior.coords)[:-1]]
            if unit.z_max is not None and (appliance.z_max is None or unit.z_max > appliance.z_max):
                appliance.z_max = unit.z_max
            items.remove(unit)
            own = union
            merged.append({"appliance": appliance.id, "removed": unit.id, "label": unit.label, "cover": round(cover, 3)})
    return merged
