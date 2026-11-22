"""Structure of one indoor level beyond plain walls: wall / room heights, columns, beams and stairs.

Two hooks of the clean plan (``samcloud.ortho.plan``) use this module:

1. :func:`analyse_plan_structure` - after the rooms are final and before the DXF result is built: streams the classified
   points once, measures every room's floor / ceiling and detects columns, beams and stairs.
   :meth:`StructureAnalysis.add_dxf_entities` draws columns (A-COLS, filled) and beams (S-BEAM, dashed overhead lines).
2. :meth:`StructureAnalysis.apply_to_plan` - on the finished ``BuildingPlan``: wall heights, room floor / ceiling and the
   ``columns`` / ``beams`` / ``stairs`` lists of ``plan_model.json``.

Both are never fatal: without points (or on any error) the plan stays as before (storey height everywhere).
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from shapely.geometry import LineString, Point, Polygon
from shapely.ops import unary_union

from .beams import Beam, detect_beams, merge_collinear_beams
from .columns import Column, detect_columns
from .common import LevelPoints, StructureSettings, load_level_points, load_structure_settings, ring_local, to_metric
from .levels import RoomLevel, WallHeight, room_levels, summarise_levels, wall_heights
from .stairs import Stair, detect_stairs

_LOGGER = logging.getLogger(__name__)
_FURNITURE_LAYERS = ("A-FURN", "A-EINB", "P-SANI")


@dataclass(slots=True)
class StructureAnalysis:
    """Metric results of one level plus what is needed to transfer them into the plan model (local units)."""

    units_per_metre: float
    floor_z: float  # local units (level floor)
    level_ceiling_m: float
    rooms: list[Polygon]  # metric, in the order of the plan rooms
    levels: list[RoomLevel]
    columns: list[Column] = field(default_factory=list)
    beams: list[Beam] = field(default_factory=list)
    stairs: list[Stair] = field(default_factory=list)
    rejected: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    points: LevelPoints | None = None
    settings: StructureSettings = field(default_factory=StructureSettings)
    wall_report: list[dict[str, Any]] = field(default_factory=list)

    # ------------------------------------------------------------------------------------------------- local units
    def _z_local(self, metres: float) -> float:
        return float(self.floor_z + metres * self.units_per_metre)

    # ----------------------------------------------------------------------------------------------------- DXF
    def add_dxf_entities(self, layers: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
        """Columns on A-COLS (outline + solid fill), beams on S-BEAM (both long edges, dashed by the layer); returns metrics."""
        upm = self.units_per_metre
        columns = layers.setdefault("A-COLS", [])
        for column in self.columns:
            ring = ring_local(column.footprint, upm)
            if column.shape == "circle":
                columns.append({"type": "circle", "center": [column.center[0] * upm, column.center[1] * upm], "radius": column.width_m / 2.0 * upm, "_structure": "column"})
            else:
                columns.append({"type": "polyline", "points": ring, "closed": True, "holes": [], "_structure": "column"})
            columns.append({"type": "hatch", "points": ring, "holes": [], "_structure": "column"})
        beams = layers.setdefault("S-BEAM", [])
        for beam in self.beams:
            (ax, ay), (bx, by) = beam.axis
            direction = np.asarray((bx - ax, by - ay))
            direction = direction / max(float(np.linalg.norm(direction)), 1e-12)
            normal = np.asarray((-direction[1], direction[0])) * beam.width_m / 2.0
            for sign in (-1.0, 1.0):
                start = (np.asarray((ax, ay)) + sign * normal) * upm
                end = (np.asarray((bx, by)) + sign * normal) * upm
                beams.append({"type": "line", "start": [float(start[0]), float(start[1])], "end": [float(end[0]), float(end[1])], "_structure": "beam"})
        return self.metrics()

    def metrics(self) -> dict[str, Any]:
        return {
            "columns": len(self.columns), "beams": len(self.beams), "stairs": len(self.stairs),
            "rooms": summarise_levels(self.levels),
            "rejected": {key: len(value) for key, value in self.rejected.items()},
        }

    # ------------------------------------------------------------------------------------------------ plan model
    def apply_to_plan(self, plan: Any) -> dict[str, Any]:
        """Write wall heights, room floor / ceiling, columns, beams and stairs into a ``BuildingPlan`` (in place)."""
        upm = self.units_per_metre
        matched = self._match_rooms(plan.rooms)
        for room, index in zip(plan.rooms, matched, strict=True):
            if index is None:
                continue
            level = self.levels[index]
            if level.floor_m is not None:
                room.floor_z = self._z_local(level.floor_m)
                room.floor_source = level.floor_source
            if level.ceiling_m is not None and level.ceiling_source != "level":
                room.ceiling_z = self._z_local(level.ceiling_m)
                room.ceiling_source = level.ceiling_source
                if level.ceiling_plane is not None:
                    a, b, c = level.ceiling_plane
                    # metric z = a x_m + b y_m + c  ->  local z = a x + b y + (floor_z + c * upm)
                    room.ceiling_plane = [float(a), float(b), self._z_local(c)]
                if level.ceiling_planes:
                    room.ceiling_planes = [[float(a), float(b), self._z_local(c)] for a, b, c in level.ceiling_planes]
                    room.ceiling_envelope = level.ceiling_envelope
        ordered_levels = [self.levels[index] if index is not None else RoomLevel() for index in matched]
        ordered_rooms = [self.rooms[index] if index is not None else to_metric(Polygon(room.polygon["exterior"]), upm) for room, index in zip(plan.rooms, matched, strict=True)]
        if self.points is not None and plan.walls:
            spans = [
                (np.asarray(wall.centreline[0], dtype=float) / upm, np.asarray(wall.centreline[-1], dtype=float) / upm, float(wall.thickness_m))
                for wall in plan.walls
            ]
            heights = wall_heights(self.points, spans, ordered_rooms, ordered_levels, self.level_ceiling_m, self.settings)
            self.wall_report = []
            for wall, height in zip(plan.walls, heights, strict=True):
                if height.height_m is not None:
                    wall.height_m = round(float(height.height_m), 4)
                wall.height_source = height.source
                self.wall_report.append({
                    "id": wall.id, "height_m": None if height.height_m is None else round(height.height_m, 3), "source": height.source,
                    "top_m": None if height.top_m is None else round(height.top_m, 3), "reference_m": None if height.reference_m is None else round(height.reference_m, 3),
                    "covered": height.covered, "stations": height.stations,
                })
        plan.columns = [self._column_dict(index, column) for index, column in enumerate(self.columns, start=1)]
        plan.beams = [self._beam_dict(index, beam) for index, beam in enumerate(self.beams, start=1)]
        plan.stairs = [self._stair_dict(index, stair) for index, stair in enumerate(self.stairs, start=1)]
        report = {**self.metrics(), "walls": self.wall_report, "rejected_detail": {key: value[:20] for key, value in self.rejected.items()}}
        plan.source["structure"] = report
        return report

    def _match_rooms(self, plan_rooms: list[Any]) -> list[int | None]:
        upm = self.units_per_metre
        result: list[int | None] = []
        for position, room in enumerate(plan_rooms):
            try:
                polygon = to_metric(Polygon(room.polygon["exterior"], room.polygon.get("holes") or None), upm).buffer(0)
            except (ValueError, TypeError):
                result.append(None)
                continue
            if position < len(self.rooms) and self.rooms[position].buffer(0).intersection(polygon).area >= 0.8 * max(polygon.area, 1e-9):
                result.append(position)
                continue
            scores = [candidate.buffer(0).intersection(polygon).area / max(candidate.union(polygon).area, 1e-9) for candidate in self.rooms]
            best = int(np.argmax(scores)) if scores else -1
            result.append(best if best >= 0 and scores[best] >= 0.6 else None)
        return result

    def _column_dict(self, index: int, column: Column) -> dict[str, Any]:
        upm = self.units_per_metre
        return {
            "id": f"C{index:03d}", "shape": column.shape, "center": [column.center[0] * upm, column.center[1] * upm],
            "footprint": ring_local(column.footprint, upm), "width_m": round(column.width_m, 4), "depth_m": round(column.depth_m, 4),
            "diameter_m": round(column.width_m, 4) if column.shape == "circle" else None, "angle": round(column.angle_rad, 6),
            "z_min": self._z_local(column.z_bottom_m), "z_max": self._z_local(column.z_top_m),
            "residual_m": round(column.residual_m, 4), "confidence": column.confidence, "evidence": column.evidence,
        }

    def _beam_dict(self, index: int, beam: Beam) -> dict[str, Any]:
        upm = self.units_per_metre
        return {
            "id": f"B{index:03d}", "axis": [[beam.axis[0][0] * upm, beam.axis[0][1] * upm], [beam.axis[1][0] * upm, beam.axis[1][1] * upm]],
            "width_m": round(beam.width_m, 4), "depth_m": round(beam.depth_m, 4), "footprint": ring_local(beam.footprint, upm),
            "z_min": self._z_local(beam.z_bottom_m), "z_max": self._z_local(beam.z_top_m), "confidence": beam.confidence, "evidence": beam.evidence,
        }

    def _stair_dict(self, index: int, stair: Stair) -> dict[str, Any]:
        upm = self.units_per_metre
        return {
            "id": f"ST{index:03d}", "start": [stair.start[0] * upm, stair.start[1] * upm], "direction_deg": round(stair.direction_deg, 4),
            "width_m": round(stair.width_m, 4), "riser_count": stair.riser_count, "riser_height_m": round(stair.riser_height_m, 4),
            "tread_depth_m": round(stair.tread_depth_m, 4), "from_z": self._z_local(stair.from_z_m), "to_z": self._z_local(stair.to_z_m),
            "footprint": ring_local(stair.footprint, upm), "confidence": stair.confidence, "evidence": stair.evidence,
        }


def analyse_structure(
    points: LevelPoints, rooms: list[Polygon], walls: list[Polygon], openings: list[Polygon], furniture: list[Polygon],
    units_per_metre: float, floor_z: float, level_ceiling_m: float, settings: StructureSettings = StructureSettings(),
    instances: list[dict[str, Any]] | None = None,
) -> StructureAnalysis:
    """All detectors on metric inputs (rooms / walls / openings / furniture as metric polygons)."""
    levels = room_levels(points, rooms, settings)
    wall_union = unary_union(walls) if walls else None
    columns, rejected_columns = detect_columns(points, rooms, levels, wall_union, openings, furniture, level_ceiling_m, settings, instances)
    beams, rejected_beams = detect_beams(points, rooms, levels, wall_union, [column.footprint for column in columns], level_ceiling_m, settings)
    beams = merge_collinear_beams(beams)
    for column in columns:  # a column under a beam ends at the beam's underside
        centre = Point(*column.center)
        below = [beam.z_bottom_m for beam in beams if beam.footprint.buffer(0.05).contains(centre)]
        if below and min(below) < column.z_top_m:
            column.z_top_m = float(min(below))
            column.evidence["top"] = "beam_underside"
    stairs, rejected_stairs = detect_stairs(points, furniture, settings)
    return StructureAnalysis(
        units_per_metre=float(units_per_metre), floor_z=float(floor_z), level_ceiling_m=float(level_ceiling_m), rooms=rooms, levels=levels,
        columns=columns, beams=beams, stairs=stairs, rejected={"columns": rejected_columns, "beams": rejected_beams, "stairs": rejected_stairs},
        points=points, settings=settings,
    )


def _opening_polygon(opening: Any, upm: float) -> Polygon | None:
    try:
        centre = np.asarray(opening.center, dtype=float)[:2] / upm
        axis = np.asarray((math.cos(float(opening.axis_angle)), math.sin(float(opening.axis_angle))))
        normal = np.asarray((-axis[1], axis[0]))
        half_w = float(opening.width) / upm / 2.0
        half_t = max(float(getattr(opening, "wall_thickness", 0.0)) / upm, 0.10) / 2.0
    except (TypeError, ValueError, AttributeError):
        return None
    return Polygon([tuple(centre + axis * a + normal * b) for a, b in ((-half_w, -half_t), (half_w, -half_t), (half_w, half_t), (-half_w, half_t))])


def _furniture_polygons(layers: dict[str, list[dict[str, Any]]], upm: float) -> list[Polygon]:
    from samcloud.bim.plan_model import _insert_footprint

    polygons: list[Polygon] = []
    for name in _FURNITURE_LAYERS:
        for entity in layers.get(name, []):
            polygon = None
            try:
                if entity.get("type") == "insert":
                    polygon = _insert_footprint(entity)
                elif entity.get("type") == "polyline" and entity.get("closed") and len(entity.get("points", [])) >= 3:
                    polygon = Polygon(entity["points"]).buffer(0)
            except Exception:  # noqa: BLE001 - one unusable entity never stops the analysis
                polygon = None
            if polygon is not None and not polygon.is_empty and polygon.area > 0:
                polygons.append(to_metric(polygon, upm))
    return polygons


def _instance_hulls(source: Path | str | None, frame: Any, upm: float) -> list[dict[str, Any]]:
    """Labelled scan instances next to the classified cloud (``instances.json``) as metric plan hulls; [] when absent."""
    if source is None:
        return []
    try:
        from samcloud.bim.plan_model import load_instance_hulls

        return [{"label": item["label"], "hull": to_metric(item["hull"], upm)} for item in load_instance_hulls(Path(source).with_name("instances.json"), frame)]
    except Exception:  # noqa: BLE001 - optional evidence
        return []


def analyse_plan_structure(
    source: Path | str | None, frame: Any, level: Any, units_per_metre: float, rooms: list[Polygon], walls: list[Polygon],
    layers: dict[str, list[dict[str, Any]]], openings: list[Any], cancel: Any = None, settings: StructureSettings | None = None,
) -> StructureAnalysis | None:
    """Hook of the indoor clean plan (local-unit inputs).  None when disabled, without points or on any error."""
    try:
        settings = settings or load_structure_settings()
        if not settings.enabled:
            return None
        upm = max(float(units_per_metre), 1e-9)
        storey_local = (float(level.ceiling_z) - float(level.floor_z)) if level.ceiling_z is not None else 2.5 * upm
        storey_m = storey_local / upm if storey_local > 0 else 2.5
        points = load_level_points(source, frame, float(level.floor_z), upm, storey_m, settings, cancel)
        if points is None:
            return None
        metric_rooms = [to_metric(room, upm) for room in rooms]
        metric_walls = [to_metric(wall, upm) for wall in walls if wall is not None and not wall.is_empty]
        metric_openings = [polygon for polygon in (_opening_polygon(opening, upm) for opening in openings) if polygon is not None]
        return analyse_structure(
            points, metric_rooms, metric_walls, metric_openings, _furniture_polygons(layers, upm), upm, float(level.floor_z), storey_m, settings,
            _instance_hulls(source, frame, upm),
        )
    except Exception as error:  # noqa: BLE001 - structure is an optional improvement of the plan
        from samcloud.dxf.cancel import Cancelled

        if isinstance(error, Cancelled):
            raise
        _LOGGER.warning("structure analysis failed; plan keeps storey heights and no columns / beams / stairs", exc_info=True)
        return None


def apply_plan_structure(plan: Any, analysis: StructureAnalysis | None) -> None:
    """Second hook: never fatal."""
    if analysis is None:
        return
    try:
        analysis.apply_to_plan(plan)
    except Exception:  # noqa: BLE001 - the plan model stays usable without structure data
        _LOGGER.warning("structure data could not be written into plan_model.json", exc_info=True)
    finally:
        analysis.points = None
