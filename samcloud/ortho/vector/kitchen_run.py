"""Kitchen run: a clean A-EINB countertop polygon along the walls, with appliance inserts.

Kitchen fine-label evidence (kitchen counter / cabinet, stove, cooktop, oven,
sink) is turned into a straight, L or U run of constant depth (0.60 m) whose back
edge lies flush on the wall.  Cooktop and sink blocks are placed on the run only
where an appliance label was observed; nothing is ever invented.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

import numpy as np
from shapely.geometry import LineString, Point, Polygon
from shapely.ops import unary_union

from .indoor import _uniform_insert
from .raster_grid import RasterGrid
from .symbols import CATALOGUE


@dataclass(frozen=True, slots=True)
class KitchenRunSettings:
    """Metric thresholds (scaled by units per metre at use)."""

    depth_m: float = 0.60
    wall_reach_m: float = 0.30
    strip_reach_m: float = 0.90
    min_leg_length_m: float = 0.60
    corner_reach_m: float = 0.35


@dataclass(slots=True)
class KitchenLeg:
    """One straight leg: back edge on the wall line, depth into the room."""

    origin: np.ndarray  # a point on the wall line (along coordinate 0)
    tangent: np.ndarray
    normal: np.ndarray  # pointing into the room
    start: float
    end: float
    depth: float

    def polygon(self) -> Polygon:
        a, b = self.origin + self.tangent * self.start, self.origin + self.tangent * self.end
        return Polygon([tuple(a), tuple(b), tuple(b + self.normal * self.depth), tuple(a + self.normal * self.depth)])


@dataclass(slots=True)
class KitchenRun:
    """The clean run: union polygon(s), its legs and a shape name (straight / L / U / multi)."""

    pieces: list[Polygon]
    legs: list[KitchenLeg]
    shape: str


def build_kitchen_run(
    mask: np.ndarray,
    grid: RasterGrid,
    wall_lines: Iterable[LineString],
    units_per_metre: float,
    settings: KitchenRunSettings = KitchenRunSettings(),
) -> KitchenRun | None:
    """Build the wall-aligned run for the kitchen evidence ``mask``; None when it is not wall based."""
    upm = max(float(units_per_metre), 1e-9)
    rows, cols = np.nonzero(np.asarray(mask, dtype=bool))
    if len(rows) < 8:
        return None
    points = np.asarray([grid.pixel_to_local(float(r), float(c)) for r, c in zip(rows, cols, strict=True)])
    centroid = points.mean(axis=0)
    candidates: list[KitchenLeg] = []
    for line in wall_lines:
        coords = np.asarray(line.coords, dtype=float)
        for first, second in zip(coords[:-1], coords[1:], strict=True):
            vector = second - first
            length = float(np.hypot(*vector))
            if length < 0.30 * upm:
                continue
            tangent = vector / length
            side = float(np.dot(centroid - first, np.asarray((-tangent[1], tangent[0]))))
            if abs(side) < 1e-9:
                continue
            normal = np.asarray((-tangent[1], tangent[0])) * (1.0 if side > 0.0 else -1.0)
            d = (points - first) @ normal
            s = (points - first) @ tangent
            near = (d >= -0.05 * upm) & (d <= settings.strip_reach_m * upm) & (s >= -0.10 * upm) & (s <= length + 0.10 * upm)
            if int(near.sum()) < 8 or float(np.percentile(d[near], 5)) > settings.wall_reach_m * upm:
                continue
            start = max(0.0, float(np.percentile(s[near], 1)))
            end = min(length, float(np.percentile(s[near], 99)))
            if end - start < settings.min_leg_length_m * upm:
                continue
            candidates.append(KitchenLeg(first.copy(), tangent, normal, start, end, settings.depth_m * upm))
    legs: list[KitchenLeg] = []
    for leg in sorted(candidates, key=lambda item: item.end - item.start, reverse=True):
        shape = leg.polygon()
        if any(shape.intersection(other.polygon()).area > 0.5 * shape.area for other in legs):
            continue
        legs.append(leg)
    if not legs:
        return None
    _snap_corners(legs, settings.corner_reach_m * upm)
    pieces_union = unary_union([leg.polygon() for leg in legs]).buffer(0)
    pieces = [pieces_union] if isinstance(pieces_union, Polygon) else [piece for piece in getattr(pieces_union, "geoms", ()) if isinstance(piece, Polygon)]
    pieces = [piece.simplify(1e-6 * upm, preserve_topology=True) for piece in pieces if not piece.is_empty]
    if not pieces:
        return None
    if len(pieces) > 1:
        shape_name = "multi"
    else:
        shape_name = {1: "straight", 2: "L", 3: "U"}.get(len(legs), "multi")
    return KitchenRun(pieces, legs, shape_name)


def _snap_corners(legs: list[KitchenLeg], reach: float) -> None:
    """Extend perpendicular legs to the wall corner they approach so L / U runs close cleanly."""
    for index, leg in enumerate(legs):
        for other in legs[index + 1:]:
            cross = float(leg.tangent[0] * other.tangent[1] - leg.tangent[1] * other.tangent[0])
            if abs(cross) < 0.9:
                continue
            # Intersection of the two wall lines: leg.origin + leg.tangent * a == other.origin + other.tangent * b
            matrix = np.column_stack((leg.tangent, -other.tangent))
            a, b = np.linalg.solve(matrix, other.origin - leg.origin)
            near_leg = min((abs(leg.start - a), "start"), (abs(leg.end - a), "end"))
            near_other = min((abs(other.start - b), "start"), (abs(other.end - b), "end"))
            if near_leg[0] <= reach and near_other[0] <= reach:
                setattr(leg, near_leg[1], float(a))
                setattr(other, near_other[1], float(b))


def appliance_insert(
    run: KitchenRun,
    kinds: tuple[str, ...] | str,
    appliance_points: np.ndarray,
    units_per_metre: float,
    label: str,
) -> dict[str, object] | None:
    """Catalogue insert for a labelled appliance, mounted on the run's wall edge (+Y into the room).

    ``appliance_points`` are the local coordinates of the appliance's label pixels.  The catalogue
    width nearest to the observed extent along the run is used; the block is centred on the
    observed centre, clamped to stay on the run.
    """
    upm = max(float(units_per_metre), 1e-9)
    names = (kinds,) if isinstance(kinds, str) else tuple(kinds)
    centre = np.asarray(appliance_points, dtype=float).mean(axis=0)
    leg = min(run.legs, key=lambda item: item.polygon().distance(Point(float(centre[0]), float(centre[1]))))
    along = (np.asarray(appliance_points, dtype=float) - leg.origin) @ leg.tangent
    extent_m = float(along.max() - along.min()) / upm
    entry = min((entry for name in names for entry in CATALOGUE[name]), key=lambda item: abs(item.width_m - extent_m) / item.width_m)
    half = entry.width_m * upm / 2.0
    if leg.end - leg.start < 2.0 * half:
        return None
    position = float(np.clip(float((centre - leg.origin) @ leg.tangent), leg.start + half, leg.end - half))
    x_axis = leg.tangent if float(np.dot(np.asarray((-leg.tangent[1], leg.tangent[0])), leg.normal)) > 0.0 else -leg.tangent
    insert = _uniform_insert(entry, leg.origin + leg.tangent * position, x_axis, upm, {"attributes": {"LABEL": label}}, 0.0, True)
    insert["_catalogue"]["on_kitchen_run"] = True  # type: ignore[index]
    return insert
