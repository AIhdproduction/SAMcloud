"""QA: the wall above and beside an opening is flush with the wall the opening interrupts.

An opening has no thickness of its own: lintel, sill, jambs, frame and cut take both faces of the host wall.  Three things
are checked on the plan model and on the built IFC scene, all with one tolerance (5 mm):

* ``plan``: the opening's band (``center`` +/- half ``wall_thickness_m``) is the band of its host wall piece;
* ``host``: the IFC host wall has the faces of that plan piece at the opening (a merge of wall pieces may not shift them);
* ``jamb``: the wall solid just outside each jamb has the faces of the lintel, at the height of the lintel, of the opening
  and of the sill.  A wall that runs on from the jamb across the opening's wall is a corner, not a step.  A short return
  (up to ``corner_reach_m``) and any parallel piece with other faces is a step.

Separately listed under ``blocked_spans`` (no lintel step, so no error): an opening that is wider than the room between two
crossing walls, so a crossing wall's end stands in or directly before its last centimetres.  That is a width of the opening
detection; the check never narrows an opening.

A step between two pieces whose faces were both measured (evidence ``face`` / ``depth``) and differ by at least
``proven_step_m`` is a real reveal: it is listed under ``proven`` and is no error.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from shapely.geometry import LineString, Point, Polygon
from shapely.ops import unary_union

from samcloud.bim.core.model import OpeningElement, WallElement

TOLERANCE_M = 0.005
_MEASURED = frozenset({"face", "depth"})
_PROBE_M = 0.002  # the profile is taken this far outside the jamb
_PARALLEL_SINE = math.sin(math.radians(10.0))


def _band(points: np.ndarray, thickness: float, at: np.ndarray, normal: np.ndarray) -> tuple[float, float]:
    """Offsets (along ``normal``, from ``at``) of both faces of a wall centreline at ``at``."""
    direction = (points[-1] - points[0]) / max(float(np.linalg.norm(points[-1] - points[0])), 1e-12)
    across = np.asarray((-direction[1], direction[0]))
    middle = float((points[0] - at) @ across) / (float(across @ normal) or 1.0)
    return middle - thickness / 2.0, middle + thickness / 2.0


def _intervals(solid: Any, origin: np.ndarray, normal: np.ndarray, low: float, high: float) -> list[tuple[float, float]]:
    hit = solid.intersection(LineString([tuple(origin + normal * low), tuple(origin + normal * high)]))
    found: list[tuple[float, float]] = []
    for part in getattr(hit, "geoms", [hit]):
        if part.is_empty or part.geom_type != "LineString":
            continue
        values = [float((np.asarray(point) - origin) @ normal) for point in part.coords]
        found.append((min(values), max(values)))
    merged: list[tuple[float, float]] = []
    for first, last in sorted(found):
        if merged and first <= merged[-1][1] + 1e-4:
            merged[-1] = (merged[-1][0], max(merged[-1][1], last))
        else:
            merged.append((first, last))
    return merged


def check_opening_flush(
    plan: Any, scene: Any, geometry: Any, tolerance_m: float = TOLERANCE_M, corner_reach_m: float = 0.15, proven_step_m: float = 0.06,
) -> dict[str, Any]:
    """``{"ok", "tolerance_m", "checked", "max_step_m", "steps", "proven", "open_jambs", "blocked_spans"}``; every step names opening, place and size.

    ``geometry`` is ``compute_scene_geometry(scene)``.  ``open_jambs`` are jambs with no wall beside them (the opening ends
    the wall); nothing is there to be flush with.
    """
    upm = float(plan.units_per_metre) or 1.0
    openings = {opening.id: opening for opening in plan.openings}
    pieces = {wall.id: wall for wall in plan.walls}
    elements = sorted((e for e in scene.elements.values() if isinstance(e, OpeningElement)), key=lambda e: e.id)
    walls = {e.id: e for e in scene.elements.values() if isinstance(e, WallElement) and e.id in geometry.walls}
    footprints = {key: Polygon(geometry.walls[key].footprint) for key in walls}
    cuts: list[tuple[str, float, float, Polygon]] = []  # (host, z from, z to, plan rectangle)
    spans: dict[str, tuple[float, float, float, float]] = {}
    for element in elements:
        wall = geometry.walls.get(element.params.host_wall_id)
        if wall is None:
            continue
        s0, s1 = element.params.position_along_axis - element.params.width / 2.0, element.params.position_along_axis + element.params.width / 2.0
        z0 = wall.base_z + element.params.sill_height
        z1 = z0 + element.params.height
        margin = 1e-4  # only in depth, as the IFC void
        rectangle = Polygon([wall.point(s0, wall.offset_right - margin), wall.point(s1, wall.offset_right - margin), wall.point(s1, wall.offset_left + margin), wall.point(s0, wall.offset_left + margin)])
        cuts.append((wall.element_id, z0, z1, rectangle))
        spans[element.id] = (s0, s1, z0, z1)

    steps: list[dict[str, Any]] = []
    proven: list[dict[str, Any]] = []
    open_jambs: list[dict[str, Any]] = []
    blocked_spans: list[dict[str, Any]] = []
    checked: set[str] = set()
    solids: dict[float, Any] = {}

    def solid_at(z: float) -> Any:
        key = round(z, 4)
        if key not in solids:
            parts = []
            for wall_id, footprint in footprints.items():
                body = geometry.walls[wall_id]
                if not body.base_z <= z <= body.top_z:
                    continue
                for host, z0, z1, rectangle in cuts:
                    if host == wall_id and z0 < z < z1:
                        footprint = footprint.difference(rectangle)
                parts.append(footprint)
            solids[key] = unary_union(parts) if parts else Polygon()
        return solids[key]

    def add(opening: Any, where: str, size: float, real: bool = False) -> None:
        (proven if real else steps).append({"opening": opening.id, "kind": opening.kind, "where": where, "step_m": round(float(size), 4)})

    for element in elements:
        opening = openings.get(element.tag)
        if opening is None or element.id not in spans:
            continue
        checked.add(opening.id)
        wall = geometry.walls[element.params.host_wall_id]
        direction, normal = np.asarray(wall.direction), np.asarray(wall.normal)
        low, high = wall.offset_right, wall.offset_left
        s0, s1, z0, z1 = spans[element.id]
        axis_point = np.asarray(wall.point((s0 + s1) / 2.0, 0.0))
        piece = pieces.get(str(opening.host_wall_id)) if opening.host_wall_id else None
        if piece is not None and piece.thickness_m > 0.0:
            line = np.asarray(piece.centreline, dtype=float) / upm
            along = (line[-1] - line[0]) / max(float(np.linalg.norm(line[-1] - line[0])), 1e-12)
            if abs(float(along[0] * direction[1] - along[1] * direction[0])) <= _PARALLEL_SINE:
                piece_band = _band(line, float(piece.thickness_m), axis_point, normal)
                size = max(abs(piece_band[0] - low), abs(piece_band[1] - high))
                if size > tolerance_m:
                    add(opening, "host", size)
                centre = np.asarray(opening.center, dtype=float) / upm
                across = np.asarray((-along[1], along[0]))
                half = float(piece.thickness_m) / 2.0
                offset = float((centre - line[0]) @ across)
                own = float(opening.wall_thickness_m) / 2.0
                size = max(abs(offset - own + half), abs(offset + own - half))
                if size > tolerance_m:
                    add(opening, "plan", size)
        levels = [("jamb", (z0 + z1) / 2.0)]
        if wall.top_z - z1 > 2.0 * tolerance_m:
            levels.append(("lintel", z1 + tolerance_m))
        if z0 - wall.base_z > 2.0 * tolerance_m:
            levels.append(("sill", z0 - tolerance_m))
        span = Polygon([
            wall.point(s0 + tolerance_m, low - tolerance_m), wall.point(s1 - tolerance_m, low - tolerance_m),
            wall.point(s1 - tolerance_m, high + tolerance_m), wall.point(s0 + tolerance_m, high + tolerance_m),
        ])
        for key, body in geometry.walls.items():
            if key == wall.element_id or key not in footprints or not footprints[key].intersects(span):
                continue
            if abs(float(np.asarray(body.direction) @ normal)) <= _PARALLEL_SINE:
                continue  # parallel pieces are judged by the jamb profiles and the overlap check
            part = footprints[key].intersection(span)
            if part.area <= tolerance_m * tolerance_m:
                continue
            x0, y0, x1, y1 = part.bounds
            along = [(x - axis_point[0]) * direction[0] + (y - axis_point[1]) * direction[1] for x, y in ((x0, y0), (x0, y1), (x1, y0), (x1, y1))]
            depth = min((s1 - s0) / 2.0 - min(along), max(along) + (s1 - s0) / 2.0)  # from the nearer jamb into the span
            blocked_spans.append({"opening": opening.id, "kind": opening.kind, "wall": walls[key].name, "overlap_m": round(float(depth), 4)})
        reach = max(high - low, 0.45) + corner_reach_m
        for jamb, s, sign in (("start", s0, -1.0), ("end", s1, 1.0)):
            origin = np.asarray(wall.point(s, 0.0)) + direction * sign * _PROBE_M
            worst: tuple[float, str, float, float] | None = None
            empty = 0
            for level, z in levels:
                bands = [item for item in _intervals(solid_at(z), origin, normal, low - reach, high + reach) if item[1] > low + tolerance_m and item[0] < high - tolerance_m]
                if not bands:
                    empty += 1
                    continue
                for (first, _), (_, last) in zip(bands[1:], bands[:-1]):
                    if worst is None or first - last > worst[0]:
                        worst = (first - last, f"{level}_{jamb}_hole", (first + last) / 2.0, z)
                for face, outward, edge, label in ((low, -1.0, bands[0][0], "right"), (high, 1.0, bands[-1][1], "left")):
                    size = abs(edge - face)
                    if size <= tolerance_m:
                        continue
                    beyond = (edge - face) * outward > 0.0
                    if beyond and size > corner_reach_m:
                        point = Point(tuple(origin + normal * (face + outward * min(size / 2.0, 0.05))))
                        parallel = any(
                            footprints[key].covers(point) and abs(float(np.asarray(body.direction)[0] * direction[1] - np.asarray(body.direction)[1] * direction[0])) <= _PARALLEL_SINE
                            for key, body in geometry.walls.items() if key in footprints
                        )
                        if not parallel:
                            continue  # a crossing wall runs on from the jamb: a corner
                    if worst is None or size > worst[0]:
                        worst = (size, f"{level}_{jamb}_{label}", (edge + face) / 2.0, z)
            if empty == len(levels):
                open_jambs.append({"opening": opening.id, "jamb": jamb})
            if worst is not None and worst[0] > tolerance_m:
                add(opening, worst[1], worst[0], worst[0] >= proven_step_m and _both_measured(plan, piece, origin, direction, normal, low, high, sign, upm))
    return {
        "ok": not steps, "tolerance_m": tolerance_m, "checked": len(checked),
        "max_step_m": max((item["step_m"] for item in steps), default=0.0), "steps": steps, "proven": proven, "open_jambs": open_jambs,
        "blocked_spans": blocked_spans,
    }


def _both_measured(plan: Any, host: Any, origin: np.ndarray, direction: np.ndarray, normal: np.ndarray, low: float, high: float, sign: float, upm: float) -> bool:
    """Whether the host and the parallel plan piece beside the jamb both carry measured faces (a real reveal)."""
    if host is None or host.evidence not in _MEASURED:
        return False
    for wall in plan.walls:
        if wall is host or wall.thickness_m <= 0.0 or wall.evidence not in _MEASURED:
            continue
        line = np.asarray(wall.centreline, dtype=float) / upm
        along = (line[-1] - line[0]) / max(float(np.linalg.norm(line[-1] - line[0])), 1e-12)
        if abs(float(along[0] * direction[1] - along[1] * direction[0])) > _PARALLEL_SINE:
            continue
        band = LineString([tuple(line[0]), tuple(line[-1])]).buffer(float(wall.thickness_m) / 2.0, cap_style=2)
        probe = LineString([tuple(origin + direction * sign * 0.01 + normal * low), tuple(origin + direction * sign * 0.01 + normal * high)])
        if band.intersects(probe):
            return True
    return False
