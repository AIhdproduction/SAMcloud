"""Scene elements for the measured building structure of a ``BuildingPlan`` (see ``samcloud.ortho.vector.structure``).

* Room ceilings: elevation (and plane for sloped ceilings) from ``PlanRoom.ceiling_z`` / ``ceiling_plane``.
* Columns -> ``ColumnElement`` (``IfcColumn``, rectangle / circle profile) and beams -> ``BeamElement`` (``IfcBeam``):
  own scene kinds written by ``samcloud.bim.core.ifc.structure`` (never furniture).
* Stairs -> ``StairElement`` (``IfcStair`` + ``IfcStairFlight``, ``samcloud.bim.core.ifc.stairs``).

Everything here is in metres in the plan frame (local plan units divided by ``units_per_metre``).
"""

from __future__ import annotations

import math
from typing import Any, Callable


from samcloud.bim.core.model import BeamElement, BeamParams, ColumnElement, ColumnParams, ParamOrigin, Scene, StairElement
from samcloud.bim.core.model.elements import PropertyValue, StairParams
from samcloud.bim.core.model.scene import Storey
from shapely.geometry import Polygon
from samcloud.bim.plan_model import BuildingPlan, PlanRoom



def room_ceiling(room: PlanRoom, upm: float, default_elevation: float, origin_xy: tuple[float, float]) -> tuple[float, tuple[float, float, float] | None, float | None, bool]:
    """``(elevation, plane_normal, plane_offset, measured)`` of the ceiling underside of one room (metres, absolute).

    ``origin_xy`` is a plan point of the room (metres) where the elevation of a sloped ceiling is evaluated.
    """
    if room.ceiling_z is None or room.ceiling_source == "level":
        return default_elevation, None, None, False
    if room.ceiling_plane is not None and len(room.ceiling_plane) == 3:
        a, b, c = (float(value) for value in room.ceiling_plane)
        # local z = a x + b y + c  ->  metres: z_m = a x_m + b y_m + c / upm  ->  n . p = offset with n = (-a, -b, 1)
        offset = c / upm
        elevation = a * origin_xy[0] + b * origin_xy[1] + offset
        return float(elevation), (-a, -b, 1.0), float(offset), True
    return float(room.ceiling_z) / upm, None, None, True


def _plane_metric(plane: list[float], upm: float) -> tuple[float, float, float]:
    a, b, c = (float(value) for value in plane)
    return a, b, c / upm


def room_ceiling_pieces(
    room: PlanRoom, upm: float, default_elevation: float, shell: Polygon,
) -> list[tuple[Polygon, float, tuple[float, float, float] | None, float | None, bool]]:
    """Ceiling pieces of one room shell (metres): ``[(outline, elevation, plane_normal, plane_offset, measured)]``.

    A two-plane ceiling (``room.ceiling_planes``) is split along the line where both planes meet (ridge for the lower
    envelope, valley for the upper); each part carries its own plane.  Otherwise one piece (flat, one plane or storey).
    """
    planes = [_plane_metric(plane, upm) for plane in (room.ceiling_planes or []) if len(plane) == 3]
    if room.ceiling_source != "level" and room.ceiling_z is not None and len(planes) == 2:
        (a0, b0, c0), (a1, b1, c1) = planes
        nx, ny, d = a0 - a1, b0 - b1, c0 - c1  # plane 0 is lower where nx x + ny y + d <= 0
        if math.hypot(nx, ny) > 1e-9:
            x0, y0, x1, y1 = shell.bounds
            size = 4.0 * max(x1 - x0, y1 - y0, 1.0)
            cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
            length = math.hypot(nx, ny)
            ux, uy = nx / length, ny / length  # towards the side where plane 0 is higher
            # closest point of the split line to the shell centre
            t = -(nx * cx + ny * cy + d) / (length * length)
            px, py = cx + nx * t, cy + ny * t
            vx, vy = -uy, ux
            lower_side = Polygon([
                (px + vx * size, py + vy * size), (px - vx * size, py - vy * size),
                (px - vx * size - ux * size, py - vy * size - uy * size), (px + vx * size - ux * size, py + vy * size - uy * size),
            ])
            first = shell.intersection(lower_side)  # plane 0 is the lower one here
            second = shell.difference(lower_side)
            if room.ceiling_envelope == "max":
                first, second = second, first
            pieces = []
            for geometry, (a, b, c) in ((first, planes[0]), (second, planes[1])):
                parts = [geometry] if geometry.geom_type == "Polygon" else [part for part in getattr(geometry, "geoms", []) if part.geom_type == "Polygon"]
                for part in parts:
                    if part.is_empty or part.area < 0.05:
                        continue
                    anchor = part.representative_point()
                    pieces.append((part, float(a * anchor.x + b * anchor.y + c), (-a, -b, 1.0), float(c), True))
            if len(pieces) >= 2:
                return pieces
    anchor = shell.representative_point()
    elevation, normal, offset, measured = room_ceiling(room, upm, default_elevation, (anchor.x, anchor.y))
    return [(shell, elevation, normal, offset, measured)]


def add_structure_elements(
    plan: BuildingPlan, scene: Scene, storey: Storey, building_id: str, uid: Callable[[str, str], str], elevation: float, height: float,
    warnings: list[str],
) -> dict[str, int]:
    """Add columns, beams and stairs of the plan to the scene; returns how many of each were added."""
    upm = plan.units_per_metre or 1.0
    counts = {"columns": 0, "beams": 0, "stairs": 0}
    for column in plan.columns:
        try:
            z0 = float(column["z_min"]) / upm if column.get("z_min") is not None else elevation
            z1 = float(column["z_max"]) / upm if column.get("z_max") is not None else elevation + height
            shape = "circle" if str(column.get("shape") or "rect") == "circle" else "rect"
            width = float(column.get("diameter_m") or column["width_m"]) if shape == "circle" else float(column["width_m"])
            params = ColumnParams(
                center=(float(column["center"][0]) / upm, float(column["center"][1]) / upm), profile=shape, width=width,
                depth=None if shape == "circle" else float(column.get("depth_m") or width), rotation=math.degrees(float(column.get("angle") or 0.0)),
                base_z=z0, height=z1 - z0,
            )
        except (KeyError, TypeError, ValueError) as error:
            warnings.append(f"column {column.get('id')}: invalid parameters ({error}), skipped")
            continue
        scene.add(ColumnElement(
            id=uid("column", str(column["id"])), name=f"Column {column['id']}", tag=str(column["id"]), storey_id=storey.id, building_id=building_id,
            params=params, param_origin={"center": ParamOrigin.FITTED, "width": ParamOrigin.FITTED, "depth": ParamOrigin.FITTED, "height": ParamOrigin.MEASURED},
            properties={
                "ProfileFitResidual": PropertyValue(value=float(column.get("residual_m") or 0.0), value_type="number"),
                "MeasuredConfidence": PropertyValue(value=float(column.get("confidence") or 0.0), value_type="number"),
            },
        ))
        counts["columns"] += 1
    for beam in plan.beams:
        try:
            z0, z1 = float(beam["z_min"]) / upm, float(beam["z_max"]) / upm
            (ax, ay), (bx, by) = beam["axis"]
            params = BeamParams(start=(float(ax) / upm, float(ay) / upm), end=(float(bx) / upm, float(by) / upm), width=float(beam["width_m"]), depth=z1 - z0, bottom_z=z0)
        except (KeyError, TypeError, ValueError) as error:
            warnings.append(f"beam {beam.get('id')}: invalid parameters ({error}), skipped")
            continue
        scene.add(BeamElement(
            id=uid("beam", str(beam["id"])), name=f"Beam {beam['id']}", tag=str(beam["id"]), storey_id=storey.id, building_id=building_id,
            params=params, param_origin={"start": ParamOrigin.FITTED, "end": ParamOrigin.FITTED, "width": ParamOrigin.MEASURED, "depth": ParamOrigin.MEASURED, "bottom_z": ParamOrigin.MEASURED},
            properties={"MeasuredConfidence": PropertyValue(value=float(beam.get("confidence") or 0.0), value_type="number")},
        ))
        counts["beams"] += 1
    for stair in plan.stairs:
        try:
            risers = int(stair["riser_count"])
            params = StairParams(
                start=(float(stair["start"][0]) / upm, float(stair["start"][1]) / upm), direction=float(stair["direction_deg"]),
                base_offset=float(stair["from_z"]) / upm - elevation, width=float(stair["width_m"]), step_count=risers,
                riser_height=float(stair["riser_height_m"]), tread_length=float(stair["tread_depth_m"]),
            )
        except (KeyError, TypeError, ValueError) as error:
            warnings.append(f"stair {stair.get('id')}: invalid parameters ({error}), skipped")
            continue
        scene.add(StairElement(
            id=uid("stair", str(stair["id"])), name=f"Stair {stair['id']}", tag=str(stair["id"]), storey_id=storey.id, building_id=building_id,
            params=params,
            param_origin={name: ParamOrigin.MEASURED for name in ("start", "direction", "width", "step_count", "riser_height", "tread_length")},
            properties={"MeasuredConfidence": PropertyValue(value=float(stair.get("confidence") or 0.0), value_type="number")},
        ))
        counts["stairs"] += 1
    return counts


def space_height_properties(room: PlanRoom, upm: float, elevation: float) -> dict[str, PropertyValue]:
    """Measured floor / ceiling information of a room as space properties."""
    props: dict[str, PropertyValue] = {
        "CeilingHeightSource": PropertyValue(value=str(room.ceiling_source), value_type="text"),
        "FloorHeightSource": PropertyValue(value=str(room.floor_source), value_type="text"),
    }
    if room.ceiling_z is not None and room.ceiling_source != "level":
        floor = float(room.floor_z) / upm if room.floor_z is not None else elevation
        props["MeasuredClearHeight"] = PropertyValue(value=round(float(room.ceiling_z) / upm - floor, 4), value_type="number")
    if room.floor_z is not None:
        props["MeasuredFloorOffset"] = PropertyValue(value=round(float(room.floor_z) / upm - elevation, 4), value_type="number")
    if room.ceiling_plane is not None:
        a, b, _ = room.ceiling_plane
        props["CeilingSlopeDeg"] = PropertyValue(value=round(math.degrees(math.atan(math.hypot(float(a), float(b)))), 3), value_type="number")
    return props
