"""CAD kitchen: the wall-fitted counter run becomes a drawable kitchen plan.

Runs after ``plan_polish.fit_kitchen_to_walls`` (legs flush on the final walls, one depth).  From the measured
heights and the appliance labels it

* extends a wall-backed leg along its wall only as far as counter-height surfaces (``counter_min_m`` ..
  ``counter_max_m``) are really measured in the front part of the leg (never into a wall or out of the room);
* finds a raised bar on a free-standing leg (peninsula): a band beside it standing above the counter top up to
  ``bar_max_m`` along most of its length becomes part of the counter, the step is drawn as an inner line;
* re-mounts the cooktop and the sink ON the counter at the position of their labels at counter height (a block
  centred in the counter depth and lying fully inside it; a sink in a chamfered inner corner sits on the
  diagonal);
* marks tall units at a leg end (full-depth evidence above ``tall_min_m``) as full-depth rectangles with a
  diagonal cross and "KS" (fridge label) or "HS" (unless the wall stage already resolved the unit there); a leg
  end within ``tall_snap_m`` of a tall unit's side reaches it;
* gives upper cabinets (A-OBEN, dashed) their diagonal cross;
* absorbs traced furniture that stands on the final counter or bar.

``counter_lines`` finally turns every run outline into open front / end lines: the side lying on a wall face is
the wall's line already.  All thresholds are metric and scaled by ``units_per_metre``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import math
from typing import Any

import numpy as np
import shapely
from shapely import affinity
from shapely.geometry import GeometryCollection, LineString, MultiLineString, Point, Polygon, box
from shapely.ops import linemerge, unary_union

from .indoor import _uniform_insert
from .plan_polish import _LegBox, _closed_polyline, _furniture_off_counter, _largest, _points, _room_of
from .symbols import CATALOGUE


@dataclass(frozen=True, slots=True)
class KitchenLayoutSettings:
    """Metric thresholds of the CAD kitchen (config ``plan_polish:`` keys ``kitchen_layout_*``)."""

    enabled: bool = True
    counter_min_m: float = 0.75
    counter_max_m: float = 1.05
    front_band_m: float = 0.30
    sample_m: float = 0.02
    extend_step_m: float = 0.05
    extend_max_m: float = 2.0
    extend_gap_m: float = 0.15
    extend_share: float = 0.35
    wall_touch_m: float = 0.05
    bar_above_m: float = 0.08
    bar_max_m: float = 1.30
    bar_share: float = 0.45
    bar_min_depth_m: float = 0.15
    bar_max_depth_m: float = 0.60
    bar_min_length_share: float = 0.50
    appliance_min_area_m2: float = 0.004
    appliance_margin_m: float = 0.02
    tall_min_m: float = 1.60
    tall_share: float = 0.50
    tall_min_length_m: float = 0.45
    tall_snap_m: float = 0.30


SINK_KINDS = ("KITCHEN_SINK",)
COOKTOP_KINDS = ("COOKTOP",)


def kitchen_cad_layout(
    layers: dict[str, list[dict[str, Any]]],
    rooms: list[Polygon],
    walls: Any,
    theta: float,
    units_per_metre: float,
    height_of: Callable[[np.ndarray], np.ndarray] | None,
    appliances: dict[str, np.ndarray] | None = None,
    settings: KitchenLayoutSettings = KitchenLayoutSettings(),
) -> dict[str, Any]:
    """Turn the fitted run(s) on A-EINB into a CAD kitchen (see the module docstring); returns statistics.

    ``appliances`` maps a catalogue kind (``COOKTOP``, ``KITCHEN_SINK``) or ``REFRIGERATOR`` to the local points of
    its label pixels measured at counter height (fridge: any height).
    """
    upm = max(float(units_per_metre), 1e-9)
    stats: dict[str, Any] = {"kitchen_legs_extended_m": 0.0, "kitchen_bars": 0, "kitchen_bar_depth_m": None, "kitchen_appliances_mounted": [],
                             "kitchen_tall_units": 0, "kitchen_upper_crosses": 0, "kitchen_layout_absorbed": 0}
    if not settings.enabled:
        return stats
    appliances = appliances or {}
    local_walls = affinity.rotate(walls, -theta, origin=(0.0, 0.0), use_radians=True) if walls is not None and not walls.is_empty else None
    counters: list[Polygon] = []
    for entity in layers.get("A-EINB", []):
        if not _closed_polyline(entity) or not entity.get("_run_legs"):
            continue
        legs = [leg for leg in (_LegBox.of(Polygon(points), theta) for points in entity["_run_legs"] if len(points) >= 3) if leg is not None]
        if not legs:
            continue
        room = _room_of(Polygon(entity["points"]).buffer(0), rooms) if rooms else None
        local_room = affinity.rotate(room, -theta, origin=(0.0, 0.0), use_radians=True) if room is not None else None
        for leg in legs:
            leg.back = _back_side(leg, local_walls, settings.wall_touch_m * upm)
        frame = _Frame(theta, height_of)
        if height_of is not None:
            for leg in legs:
                if leg.back is not None:
                    stats["kitchen_legs_extended_m"] += _extend_leg(leg, legs, local_walls, local_room, frame, settings, upm) / upm
        # a counter ends at the side of a tall unit next to it (the units abut)
        units = [affinity.rotate(Polygon(item["points"]), -theta, origin=(0.0, 0.0), use_radians=True) for item in layers.get("A-EINB", []) if _closed_polyline(item) and item.get("_einb_kind") == "tall_unit"]
        if units:
            local_units = unary_union(units)
            for leg in legs:
                if leg.back is not None:
                    leg.snap_ends(local_units, settings.tall_snap_m * upm)
        bars: list[tuple[Polygon, LineString]] = []
        if height_of is not None:
            top = _counter_top(legs, frame, settings, upm)
            for leg in legs:
                if leg.back is None:
                    bars.extend(_bars_of(leg, legs, local_walls, local_room, frame, top, settings, upm))
        chamfers = [Polygon(points) for points in entity.get("_chamfers", []) if len(points) >= 3]
        shapes = [leg.polygon(theta) for leg in legs] + [affinity.rotate(bar, theta, origin=(0.0, 0.0), use_radians=True) for bar, _ in bars] + chamfers
        union = unary_union(shapes)
        if walls is not None and not walls.is_empty:
            union = union.difference(walls)
        if room is not None:
            union = union.intersection(room.buffer(1e-6 * upm, join_style=2))
        shape = _largest(union.buffer(0.002 * upm, join_style=2).buffer(-0.002 * upm, join_style=2))
        if shape is None:
            continue
        shape = shape.simplify(0.002 * upm, preserve_topology=True)
        entity["points"] = _points(shape)
        entity["_run_legs"] = [_points(leg.polygon(theta)) for leg in legs]
        counters.append(shape)
        for bar, step in bars:
            stats["kitchen_bars"] += 1
            depth = (bar.bounds[3] - bar.bounds[1]) if (bar.bounds[2] - bar.bounds[0]) >= (bar.bounds[3] - bar.bounds[1]) else (bar.bounds[2] - bar.bounds[0])
            stats["kitchen_bar_depth_m"] = round(depth / upm, 3)
            line = affinity.rotate(step, theta, origin=(0.0, 0.0), use_radians=True).intersection(shape.buffer(0.001 * upm))
            for part in _line_parts(line):
                if part.length >= 0.10 * upm:
                    layers.setdefault("A-FURN-DETL", []).append({"type": "polyline", "points": [[float(x), float(y)] for x, y in part.coords], "closed": False, "_shape_detail": True, "_kitchen_detail": "bar_step"})
        # appliances ON the counter
        layers["A-EINB"] = _mount_appliances(layers["A-EINB"], entity, legs, shape, chamfers, appliances, theta, settings, upm, stats)
        if height_of is not None:
            _tall_units(layers, legs, local_walls, local_room, frame, appliances, theta, settings, upm, stats)
    if counters:
        stats["kitchen_layout_absorbed"] = _furniture_off_counter(layers, unary_union(counters), theta, upm)
    # upper cabinets: dashed outline with a diagonal cross (A-OBEN)
    crosses: list[dict[str, Any]] = []
    for entity in layers.get("A-OBEN", []):
        if not _closed_polyline(entity) or not entity.get("_upper_cabinet"):
            continue
        ring = Polygon(entity["points"]).minimum_rotated_rectangle
        if not isinstance(ring, Polygon) or ring.area <= 0.0:
            continue
        corners = list(ring.exterior.coords)[:4]
        crosses.append({"type": "polyline", "points": [list(corners[0]), list(corners[2])], "closed": False, "_upper_cross": True})
        crosses.append({"type": "polyline", "points": [list(corners[1]), list(corners[3])], "closed": False, "_upper_cross": True})
        stats["kitchen_upper_crosses"] += 1
    layers.setdefault("A-OBEN", []).extend(crosses)
    return stats


def counter_lines(layers: dict[str, list[dict[str, Any]]], walls: Any, units_per_metre: float) -> int:
    """Draw every fitted run as open front / end lines: the parts of its outline on a wall face are left out."""
    upm = max(float(units_per_metre), 1e-9)
    if walls is None or walls.is_empty:
        return 0
    band = walls.buffer(0.02 * upm, join_style=2)
    output: list[dict[str, Any]] = []
    converted = 0
    for entity in layers.get("A-EINB", []):
        if not _closed_polyline(entity) or not entity.get("_run_legs"):
            output.append(entity)
            continue
        outline = Polygon(entity["points"]).exterior
        free = outline.difference(band)
        parts = [part for part in _line_parts(linemerge(free) if isinstance(free, MultiLineString) else free) if part.length >= 0.02 * upm]
        if not parts or free.length >= 0.995 * outline.length:
            output.append(entity)  # nothing lies on a wall: keep the closed outline
            continue
        converted += 1
        meta = {key: value for key, value in entity.items() if key.startswith("_")}
        for part in parts:
            output.append({"type": "polyline", "points": [[float(x), float(y)] for x, y in part.coords], "closed": False, **meta, "_counter_front": True, "_counter_outline": entity["points"]})
    layers["A-EINB"] = output
    return converted


# --------------------------------------------------------------------------------------------- geometry helpers

@dataclass(slots=True)
class _Frame:
    theta: float
    height_of: Callable[[np.ndarray], np.ndarray] | None

    def heights(self, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        """Heights at wall-frame points (u, v) = local rotated by -theta about the origin."""
        if self.height_of is None:
            return np.full(np.shape(u), np.nan)
        cos, sin = math.cos(self.theta), math.sin(self.theta)
        u, v = np.asarray(u, dtype=float), np.asarray(v, dtype=float)
        points = np.column_stack(((u * cos - v * sin).ravel(), (u * sin + v * cos).ravel()))
        return np.asarray(self.height_of(points), dtype=float).reshape(u.shape)


def _sample_box(frame: _Frame, u0: float, u1: float, v0: float, v1: float, step: float) -> np.ndarray:
    us = np.arange(min(u0, u1) + step / 2.0, max(u0, u1), step)
    vs = np.arange(min(v0, v1) + step / 2.0, max(v0, v1), step)
    if len(us) == 0 or len(vs) == 0:
        return np.zeros((0,))
    gu, gv = np.meshgrid(us, vs)
    return frame.heights(gu, gv).ravel()


def _share(values: np.ndarray, low: float, high: float) -> float:
    if values.size == 0:
        return 0.0
    finite = np.nan_to_num(values, nan=-1e9)
    return float(np.mean((finite >= low) & (finite <= high)))


def _back_side(leg: _LegBox, local_walls: Any, reach: float) -> str | None:
    """The long edge of a leg that lies on a wall face (within ``reach``)."""
    if local_walls is None:
        return None
    best: tuple[float, str] | None = None
    for name in (("v0", "v1") if leg.horizontal else ("u0", "u1")):
        value = getattr(leg, name)
        if leg.horizontal:
            margin = 0.15 * (leg.u1 - leg.u0)
            probe = LineString([(leg.u0 + margin, value), (leg.u1 - margin, value)])
        else:
            margin = 0.15 * (leg.v1 - leg.v0)
            probe = LineString([(value, leg.v0 + margin), (value, leg.v1 - margin)])
        distance = local_walls.distance(probe)
        covered = probe.intersection(local_walls.buffer(reach)).length >= 0.5 * probe.length
        if distance <= reach and covered and (best is None or distance < best[0]):
            best = (distance, name)
    return best[1] if best is not None else None


def _blocked(local_walls: Any, local_room: Any, cell: Polygon) -> bool:
    if local_walls is not None and cell.intersection(local_walls).area >= 0.20 * cell.area:
        return True
    return local_room is not None and cell.intersection(local_room).area < 0.80 * cell.area


def _front_band(leg: _LegBox, width: float) -> tuple[float, float]:
    """Across-range (low, high) of the front band of a wall-backed leg."""
    if leg.horizontal:
        return (leg.v0, min(leg.v1, leg.v0 + width)) if leg.back == "v1" else (max(leg.v0, leg.v1 - width), leg.v1)
    return (leg.u0, min(leg.u1, leg.u0 + width)) if leg.back == "u1" else (max(leg.u0, leg.u1 - width), leg.u1)


def _end_free(leg: _LegBox, end: str, legs: list[_LegBox], local_walls: Any, reach: float) -> bool:
    """True when an end of a leg neither touches a wall nor runs into another leg."""
    value = getattr(leg, end)
    if leg.horizontal:
        probe = LineString([(value, leg.v0 + 0.1 * (leg.v1 - leg.v0)), (value, leg.v1 - 0.1 * (leg.v1 - leg.v0))])
    else:
        probe = LineString([(leg.u0 + 0.1 * (leg.u1 - leg.u0), value), (leg.u1 - 0.1 * (leg.u1 - leg.u0), value)])
    if local_walls is not None and local_walls.distance(probe) <= reach:
        return False
    return not any(other is not leg and box(other.u0, other.v0, other.u1, other.v1).buffer(reach).covers(probe) for other in legs)


def _extend_leg(leg: _LegBox, legs: list[_LegBox], local_walls: Any, local_room: Any, frame: _Frame, settings: KitchenLayoutSettings, upm: float) -> float:
    """Grow the free ends of a wall-backed leg while counter-height surfaces are measured in its front band."""
    step = settings.extend_step_m * upm
    gap_steps = max(1, int(round(settings.extend_gap_m / settings.extend_step_m)))
    reach = settings.wall_touch_m * upm
    low, high = _front_band(leg, settings.front_band_m * upm)
    grown = 0.0
    for end, sign in ((("u0", -1.0), ("u1", 1.0)) if leg.horizontal else (("v0", -1.0), ("v1", 1.0))):
        if not _end_free(leg, end, legs, local_walls, reach):
            continue
        start = getattr(leg, end)
        best = start
        misses = 0
        for k in range(int(settings.extend_max_m / settings.extend_step_m)):
            a, b = start + sign * k * step, start + sign * (k + 1) * step
            cell = box(min(a, b), low, max(a, b), high) if leg.horizontal else box(low, min(a, b), high, max(a, b))
            full = box(min(a, b), leg.v0, max(a, b), leg.v1) if leg.horizontal else box(leg.u0, min(a, b), leg.u1, max(a, b))
            if _blocked(local_walls, local_room, full):
                break
            values = _sample_box(frame, *(cell.bounds[0], cell.bounds[2], cell.bounds[1], cell.bounds[3]), settings.sample_m * upm)
            if _share(values, settings.counter_min_m * upm, settings.counter_max_m * upm) >= settings.extend_share:
                best = b
                misses = 0
            else:
                misses += 1
                if misses > gap_steps:
                    break
        if best != start:
            # an end within one step of a wall face snaps onto it
            if local_walls is not None:
                probe_end = best + sign * step
                full = box(min(best, probe_end), leg.v0, max(best, probe_end), leg.v1) if leg.horizontal else box(leg.u0, min(best, probe_end), leg.u1, max(best, probe_end))
                hit = full.intersection(local_walls)
                if not hit.is_empty and hit.area > 1e-12:
                    bounds = hit.bounds
                    best = (bounds[0] if sign > 0 else bounds[2]) if leg.horizontal else (bounds[1] if sign > 0 else bounds[3])
            grown += abs(best - start)
            setattr(leg, end, best)
    return grown


def _counter_top(legs: list[_LegBox], frame: _Frame, settings: KitchenLayoutSettings, upm: float) -> float:
    values = np.concatenate([_sample_box(frame, leg.u0, leg.u1, leg.v0, leg.v1, 0.04 * upm) for leg in legs]) if legs else np.zeros((0,))
    finite = values[np.isfinite(values) & (values >= settings.counter_min_m * upm) & (values <= settings.counter_max_m * upm)]
    return float(np.median(finite)) if finite.size else 0.90 * upm


def _bars_of(leg: _LegBox, legs: list[_LegBox], local_walls: Any, local_room: Any, frame: _Frame, top: float, settings: KitchenLayoutSettings, upm: float) -> list[tuple[Polygon, LineString]]:
    """Raised bar bands beside the long sides and the free ends of a free-standing leg (peninsula)."""
    low_h, high_h = top + settings.bar_above_m * upm, settings.bar_max_m * upm
    step = settings.sample_m * upm
    output: list[tuple[Polygon, LineString]] = []
    long_sides = (("v0", -1.0), ("v1", 1.0)) if leg.horizontal else (("u0", -1.0), ("u1", 1.0))
    extent = [leg.u0, leg.u1, leg.v0, leg.v1]
    for name, sign in long_sides:
        found = _raised_band(frame, leg, name, sign, (leg.u0, leg.u1) if leg.horizontal else (leg.v0, leg.v1), low_h, high_h, local_walls, local_room, settings, upm)
        if found is None:
            continue
        a0, a1, depth = found
        edge = getattr(leg, name)
        far = edge + sign * depth
        if leg.horizontal:
            bar = box(a0, min(edge, far), a1, max(edge, far))
            step_line = LineString([(a0, edge), (a1, edge)])
            extent[2], extent[3] = min(extent[2], far), max(extent[3], far)
        else:
            bar = box(min(edge, far), a0, max(edge, far), a1)
            step_line = LineString([(edge, a0), (edge, a1)])
            extent[0], extent[1] = min(extent[0], far), max(extent[1], far)
        output.append((bar, step_line))
    # free ends: the bar may wrap around the end of the peninsula (across the leg and its side bars)
    reach = settings.wall_touch_m * upm
    for end, sign in ((("u0", -1.0), ("u1", 1.0)) if leg.horizontal else (("v0", -1.0), ("v1", 1.0))):
        if not _end_free(leg, end, legs, local_walls, reach):
            continue
        across = (extent[2], extent[3]) if leg.horizontal else (extent[0], extent[1])
        # the top continues to the end of the counter-or-bar surface; the step is drawn only where the extension
        # is the raised bar alone
        found = _raised_band(frame, leg, end, sign, across, settings.counter_min_m * upm, high_h, local_walls, local_room, settings, upm, end_band=True)
        if found is None:
            continue
        a0, a1, depth = found
        raised_only = _raised_band(frame, leg, end, sign, across, low_h, high_h, local_walls, local_room, settings, upm, end_band=True)
        edge = getattr(leg, end)
        far = edge + sign * depth
        step_line = (LineString([(edge, leg.v0), (edge, leg.v1)]) if leg.horizontal else LineString([(leg.u0, edge), (leg.u1, edge)])) if raised_only is not None else LineString()
        if leg.horizontal:
            output.append((box(min(edge, far), a0, max(edge, far), a1), step_line))
        else:
            output.append((box(a0, min(edge, far), a1, max(edge, far)), step_line))
    return output


def _raised_band(
    frame: _Frame, leg: _LegBox, name: str, sign: float, along: tuple[float, float], low_h: float, high_h: float,
    local_walls: Any, local_room: Any, settings: KitchenLayoutSettings, upm: float, *, end_band: bool = False,
) -> tuple[float, float, float] | None:
    """(along0, along1, depth) of a raised band outside edge ``name`` of the leg, or None."""
    step = settings.sample_m * upm
    edge = getattr(leg, name)
    # the edge runs along u when it is a long side of a horizontal leg or an end of a vertical leg
    runs_along_u = leg.horizontal != end_band
    positions = np.arange(along[0] + step / 2.0, along[1], step)
    layers = np.arange(step / 2.0, settings.bar_max_depth_m * upm, step)
    if len(positions) < 3 or len(layers) < 3:
        return None
    pa, la = np.meshgrid(positions, layers)  # rows = depth layers
    across = edge + sign * la
    heights = frame.heights(pa, across) if runs_along_u else frame.heights(across, pa)
    raised = np.nan_to_num(heights, nan=-1e9)
    raised = (raised >= low_h) & (raised <= high_h)
    # coverage per depth layer, smoothed over three layers (the layer right at the edge mixes counter and bar)
    per_layer = np.convolve(raised.mean(axis=1), np.ones(3) / 3.0, mode="same")
    start = next((index for index in range(min(4, len(layers))) if per_layer[index] >= settings.bar_share), None)
    if start is None:
        return None
    depth_index = start
    while depth_index < len(layers) and per_layer[depth_index] >= settings.bar_share:
        depth_index += 1
    depth = depth_index * step
    if depth < settings.bar_min_depth_m * upm:
        return None
    covered = raised[:depth_index].mean(axis=0) >= 0.4
    if covered.mean() < settings.bar_min_length_share and not end_band:
        return None
    indices = np.flatnonzero(covered)
    if len(indices) == 0:
        return None
    a0, a1 = float(positions[indices[0]] - step / 2.0), float(positions[indices[-1]] + step / 2.0)
    if not end_band:
        # a bar that covers nearly the whole side runs to the ends of the leg
        if a0 - along[0] <= 0.15 * upm:
            a0 = along[0]
        if along[1] - a1 <= 0.15 * upm:
            a1 = along[1]
    else:
        a0, a1 = along
    far = edge + sign * depth
    cell = box(min(a0, a1), min(edge, far), max(a0, a1), max(edge, far)) if runs_along_u else box(min(edge, far), min(a0, a1), max(edge, far), max(a0, a1))
    if local_walls is not None and cell.intersection(local_walls).area >= 0.10 * cell.area:
        return None
    if local_room is not None and cell.intersection(local_room).area < 0.90 * cell.area:
        return None
    return a0, a1, depth


# --------------------------------------------------------------------------------------------- appliances

def _mount_appliances(
    einb: list[dict[str, Any]], run: dict[str, Any], legs: list[_LegBox], counter: Polygon, chamfers: list[Polygon],
    appliances: dict[str, np.ndarray], theta: float, settings: KitchenLayoutSettings, upm: float, stats: dict[str, Any],
) -> list[dict[str, Any]]:
    """Cooktop / sink inserts re-mounted on the final counter at their label position, fully inside it."""
    zone = counter.buffer(0.10 * upm)
    output: list[dict[str, Any]] = []
    existing: dict[str, list[dict[str, Any]]] = {}
    for entity in einb:
        catalogue = entity.get("_catalogue") if entity.get("type") == "insert" else None
        if catalogue and catalogue.get("kind") in {"COOKTOP", "KITCHEN_SINK"} and zone.contains(Point(entity["point"][:2])):
            existing.setdefault(str(catalogue["kind"]), []).append(entity)
            continue
        output.append(entity)
    for kind in ("KITCHEN_SINK", "COOKTOP"):
        points = np.asarray(appliances.get(kind, np.zeros((0, 2))), dtype=float).reshape(-1, 2)
        if len(points):
            points = points[shapely.contains_xy(counter.buffer(settings.appliance_margin_m * upm + 0.05 * upm), points[:, 0], points[:, 1])]
        pixel_area = (0.01 * upm) ** 2
        olds = existing.get(kind, [])
        label = str((olds[0].get("attributes") or {}).get("LABEL", "")) if olds else ""
        reach = 0.15 * upm
        if len(points) * pixel_area < settings.appliance_min_area_m2 * upm * upm:
            reach = 0.60 * upm  # no label: the old block (possibly hanging off a moved leg) only gives the position
            if not olds:
                continue
            points = np.asarray([olds[0]["point"][:2]], dtype=float)
            # the old block hangs from its mounting edge: use its centre
            footprint = _block_footprint(olds[0], upm)
            if footprint is not None:
                points = np.asarray([footprint.centroid.coords[0]], dtype=float)
        placed = _mount(kind, points, legs, counter, chamfers, theta, settings, upm, label or ("Spüle" if kind == "KITCHEN_SINK" else "Herd"), reach=reach)
        if placed is None:
            output.extend(olds)
            continue
        if any(_block_footprint(placed, upm).intersects(_block_footprint(other, upm)) for other in output if other.get("type") == "insert" and other.get("_catalogue", {}).get("on_kitchen_run")):
            output.extend(olds)
            continue
        output.append(placed)
        stats["kitchen_appliances_mounted"].append(kind)
    return output


def _block_footprint(insert: dict[str, Any], upm: float) -> Polygon | None:
    catalogue = insert.get("_catalogue")
    if not catalogue:
        return None
    width, depth = float(catalogue["width_m"]) * upm, float(catalogue["depth_m"]) * upm
    body = box(-width / 2.0, 0.0, width / 2.0, depth)
    body = affinity.rotate(body, float(insert["rotation"]), origin=(0.0, 0.0))
    return affinity.translate(body, float(insert["point"][0]), float(insert["point"][1]))


def _densest_cluster(points: np.ndarray, radius: float) -> np.ndarray:
    """The label points around the densest spot (a stray label elsewhere on the run does not pull the block)."""
    if len(points) <= 3:
        return points
    sample = points[:: max(1, len(points) // 400)]
    counts = [int(np.count_nonzero(np.hypot(*(points - point).T) <= radius)) for point in sample]
    seed = sample[int(np.argmax(counts))]
    return points[np.hypot(*(points - seed).T) <= 2.0 * radius]


def _mount(kind: str, points: np.ndarray, legs: list[_LegBox], counter: Polygon, chamfers: list[Polygon], theta: float, settings: KitchenLayoutSettings, upm: float, label: str, *, reach: float) -> dict[str, Any] | None:
    points = _densest_cluster(points, 0.25 * upm)
    centre = np.median(points, axis=0)
    entries = sorted(CATALOGUE[kind], key=lambda entry: entry.width_m)
    inside = counter.buffer(1e-4 * upm)
    # an appliance in a chamfered inner corner sits on the diagonal
    for chamfer in chamfers:
        if chamfer.buffer(0.15 * upm).contains(Point(*centre)):
            placed = _mount_on_chamfer(kind, chamfer, counter, entries, upm, label)
            if placed is not None:
                return placed
    cos, sin = math.cos(theta), math.sin(theta)
    u, v = centre[0] * cos + centre[1] * sin, -centre[0] * sin + centre[1] * cos
    backed = [leg for leg in legs if leg.back is not None]
    if not backed:
        return None

    def distance(leg: _LegBox) -> float:
        return float(box(leg.u0, leg.v0, leg.u1, leg.v1).distance(Point(u, v)))

    order = sorted(backed, key=distance)
    pu = points[:, 0] * cos + points[:, 1] * sin
    pv = -points[:, 0] * sin + points[:, 1] * cos
    for leg in order[:2]:
        if distance(leg) > reach:
            continue
        along = pu if leg.horizontal else pv
        extent_m = float(np.percentile(along, 95) - np.percentile(along, 5)) / upm if len(along) > 3 else 0.0
        position = u if leg.horizontal else v
        candidates = sorted(entries, key=lambda entry: abs(entry.width_m - extent_m))
        for entry in candidates:
            placed = _mount_on_leg(kind, entry, leg, position, legs, inside, theta, upm, label)
            if placed is not None:
                return placed
    return None


def _free_range(leg: _LegBox, legs: list[_LegBox]) -> tuple[float, float]:
    """Along-range of a leg that is not shared with a perpendicular leg (the corner squares belong to both)."""
    low, high = (leg.u0, leg.u1) if leg.horizontal else (leg.v0, leg.v1)
    for other in legs:
        if other is leg or other.horizontal == leg.horizontal:
            continue
        o_low, o_high = (other.u0, other.u1) if leg.horizontal else (other.v0, other.v1)
        across_overlap = (min(leg.v1, other.v1) - max(leg.v0, other.v0)) if leg.horizontal else (min(leg.u1, other.u1) - max(leg.u0, other.u0))
        if across_overlap <= 0.0:
            continue
        if o_low <= low + 1e-9 < o_high:
            low = max(low, o_high)
        if o_low < high - 1e-9 <= o_high:
            high = min(high, o_low)
    return low, high


def _mount_on_leg(kind: str, entry: Any, leg: _LegBox, position: float, legs: list[_LegBox], inside: Polygon, theta: float, upm: float, label: str) -> dict[str, Any] | None:
    half = entry.width_m * upm / 2.0
    low, high = _free_range(leg, legs)
    if high - low < 2.0 * half:
        low, high = (leg.u0, leg.u1) if leg.horizontal else (leg.v0, leg.v1)
    if high - low < 2.0 * half:
        return None
    position = float(np.clip(position, low + half, high - half))
    depth = leg.depth
    offset = max(0.0, (depth - entry.depth_m * upm) / 2.0)
    if leg.horizontal:
        back = leg.v1 if leg.back == "v1" else leg.v0
        normal = (0.0, -1.0) if leg.back == "v1" else (0.0, 1.0)
        origin_uv = (position, back + normal[1] * offset)
    else:
        back = leg.u1 if leg.back == "u1" else leg.u0
        normal = (-1.0, 0.0) if leg.back == "u1" else (1.0, 0.0)
        origin_uv = (back + normal[0] * offset, position)
    cos, sin = math.cos(theta), math.sin(theta)
    origin = np.asarray((origin_uv[0] * cos - origin_uv[1] * sin, origin_uv[0] * sin + origin_uv[1] * cos))
    n = np.asarray((normal[0] * cos - normal[1] * sin, normal[0] * sin + normal[1] * cos))
    x_axis = np.asarray((n[1], -n[0]))
    insert = _uniform_insert(entry, origin, x_axis, upm, {"attributes": {"LABEL": label}}, 0.0, True)
    insert["_catalogue"]["on_kitchen_run"] = True  # type: ignore[index]
    footprint = _block_footprint(insert, upm)
    if footprint is None or not inside.covers(footprint):
        return None
    return insert


def _mount_on_chamfer(kind: str, chamfer: Polygon, counter: Polygon, entries: list[Any], upm: float, label: str) -> dict[str, Any] | None:
    """A block on the diagonal front of a chamfered inner corner: front edge parallel to the diagonal, inside the counter."""
    coords = list(chamfer.exterior.coords)[:-1]
    best: tuple[float, tuple[np.ndarray, np.ndarray]] | None = None
    for index in range(len(coords)):
        a, b = np.asarray(coords[index]), np.asarray(coords[(index + 1) % len(coords)])
        vector = b - a
        angle = abs(((math.degrees(math.atan2(vector[1], vector[0])) + 45.0) % 90.0) - 45.0)
        length = float(np.hypot(*vector))
        if angle > 20.0 and length > 0.0 and (best is None or length > best[0]):
            best = (length, (a, b))
    if best is None:
        return None
    a, b = best[1]
    middle = (a + b) / 2.0
    tangent = (b - a) / float(np.hypot(*(b - a)))
    normal = np.asarray((-tangent[1], tangent[0]))
    if counter.contains(Point(*(middle + normal * 0.05 * upm))):
        normal = -normal  # +Y points out of the counter into the room
    inside = counter.buffer(1e-4 * upm)
    for entry in sorted(entries, key=lambda item: -item.width_m):
        for back_off in np.arange(0.03, 0.60, 0.01):
            origin = middle - normal * (entry.depth_m + back_off) * upm
            insert = _uniform_insert(entry, origin, np.asarray((normal[1], -normal[0])), upm, {"attributes": {"LABEL": label}}, 0.0, True)
            insert["_catalogue"]["on_kitchen_run"] = True  # type: ignore[index]
            insert["_catalogue"]["corner"] = True  # type: ignore[index]
            footprint = _block_footprint(insert, upm)
            if footprint is not None and inside.covers(footprint):
                return insert
    return None


# --------------------------------------------------------------------------------------------- tall units

def _tall_units(
    layers: dict[str, list[dict[str, Any]]], legs: list[_LegBox], local_walls: Any, local_room: Any, frame: _Frame,
    appliances: dict[str, np.ndarray], theta: float, settings: KitchenLayoutSettings, upm: float, stats: dict[str, Any],
) -> None:
    """Full-depth tall units (fridge / tall cabinet) right beyond a free end of a wall-backed leg."""
    fridge = np.asarray(appliances.get("REFRIGERATOR", np.zeros((0, 2))), dtype=float).reshape(-1, 2)
    step = settings.extend_step_m * upm
    reach = settings.wall_touch_m * upm
    cos, sin = math.cos(theta), math.sin(theta)
    for leg in legs:
        if leg.back is None:
            continue
        for end, sign in ((("u0", -1.0), ("u1", 1.0)) if leg.horizontal else (("v0", -1.0), ("v1", 1.0))):
            if not _end_free(leg, end, legs, local_walls, reach):
                continue
            start = getattr(leg, end)
            stop = start
            for k in range(int(1.5 / settings.extend_step_m)):
                a, b = start + sign * k * step, start + sign * (k + 1) * step
                full = box(min(a, b), leg.v0, max(a, b), leg.v1) if leg.horizontal else box(leg.u0, min(a, b), leg.u1, max(a, b))
                if _blocked(local_walls, local_room, full):
                    break
                values = _sample_box(frame, full.bounds[0], full.bounds[2], full.bounds[1], full.bounds[3], settings.sample_m * upm)
                if _share(values, settings.tall_min_m * upm, 1e9) < settings.tall_share:
                    break
                stop = b
            if abs(stop - start) < settings.tall_min_length_m * upm:
                continue
            rect = box(min(start, stop), leg.v0, max(start, stop), leg.v1) if leg.horizontal else box(leg.u0, min(start, stop), leg.u1, max(start, stop))
            shape = affinity.rotate(rect, theta, origin=(0.0, 0.0), use_radians=True)
            existing = [Polygon(entity["points"]) for entity in layers["A-EINB"] if _closed_polyline(entity) and entity.get("_einb_kind") == "tall_unit"]
            if any(shape.intersection(other).area > 0.30 * shape.area for other in existing):
                continue  # already resolved as a tall unit by the wall stage
            is_fridge = len(fridge) > 0 and shapely.contains_xy(shape.buffer(0.10 * upm), fridge[:, 0], fridge[:, 1]).mean() > 0.0
            layers["A-EINB"].extend(tall_unit_entities(shape, "KS" if is_fridge else "HS", upm))
            stats["kitchen_tall_units"] += 1


def tall_unit_entities(shape: Polygon, label: str, units_per_metre: float, **meta: Any) -> list[dict[str, Any]]:
    """A-EINB entities of one tall unit: the outline, a diagonal cross and the "KS" / "HS" label."""
    corners = list(shape.exterior.coords)[:4]
    centre = shape.centroid
    return [
        {"type": "polyline", "points": _points(shape), "closed": True, "holes": [], "_einb_kind": "tall_unit", "_tall_unit": label, **meta},
        {"type": "polyline", "points": [list(corners[0]), list(corners[2])], "closed": False, "_tall_unit_cross": True},
        {"type": "polyline", "points": [list(corners[1]), list(corners[3])], "closed": False, "_tall_unit_cross": True},
        {"type": "text", "text": label, "point": [float(centre.x), float(centre.y)], "height": 0.08 * units_per_metre, "rotation": 0.0, "_tall_unit_label": True},
    ]


def _line_parts(geometry: Any) -> list[LineString]:
    if geometry is None or geometry.is_empty:
        return []
    if isinstance(geometry, LineString):
        return [geometry]
    if isinstance(geometry, (MultiLineString, GeometryCollection)):
        output: list[LineString] = []
        for member in geometry.geoms:
            output.extend(_line_parts(member))
        return output
    return []
