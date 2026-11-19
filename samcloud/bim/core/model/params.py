"""Parameter schemas: unit, range and step of every element parameter (section 5.1).

The UI and the AI tools read the same schema through :func:`get_param_schema`.
Ranges are advisory limits for editing and tool calls; the pydantic models enforce
only structural validity (positive dimensions, closed outlines, and so on).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from samcloud.bim.core.model.enums import (
    DoorOperation,
    ElementKind,
    JoinEnd,
    JoinType,
    Side,
    Tristate,
    WallFunction,
    WallJustification,
    WindowOperation,
)


@dataclass(frozen=True)
class ParamSpec:
    """Description of one parameter.

    ``value_type`` is one of ``number``, ``string``, ``id``, ``enum``, ``point2``,
    ``vector3``, ``polygon``, ``polygon_list``, ``layers`` or ``flights``.
    """

    name: str
    unit: str = ""
    minimum: float | None = None
    maximum: float | None = None
    step: float | None = None
    description: str = ""
    value_type: str = "number"
    choices: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable description."""
        result: dict[str, Any] = {
            "name": self.name,
            "unit": self.unit,
            "min": self.minimum,
            "max": self.maximum,
            "step": self.step,
            "type": self.value_type,
            "description": self.description,
        }
        if self.choices:
            result["choices"] = list(self.choices)
        return result


def _num(
    name: str, unit: str, minimum: float, maximum: float, step: float, description: str
) -> ParamSpec:
    return ParamSpec(name, unit, minimum, maximum, step, description)


def _enum(name: str, choices: type[StrEnum], description: str) -> ParamSpec:
    return ParamSpec(
        name,
        description=description,
        value_type="enum",
        choices=tuple(member.value for member in choices),
    )


def _typed(name: str, value_type: str, unit: str, description: str) -> ParamSpec:
    return ParamSpec(name, unit, description=description, value_type=value_type)


_THICKNESS = (0.02, 2.0, 0.005)

PARAM_SCHEMAS: dict[ElementKind, tuple[ParamSpec, ...]] = {
    ElementKind.WALL: (
        _typed("start", "point2", "m", "Axis start point (x, y) in the local frame."),
        _typed("end", "point2", "m", "Axis end point (x, y) in the local frame."),
        _num("thickness", "m", *_THICKNESS, "Total wall thickness."),
        _num("height", "m", 0.5, 20.0, 0.01, "Wall height."),
        _num("base_offset", "m", -10.0, 10.0, 0.005, "Base elevation offset to the storey."),
        _enum("justification", WallJustification, "Axis position relative to the wall body."),
        _typed("layers", "layers", "m", "Layer set (material, thickness); sum equals thickness."),
        _enum("function", WallFunction, "Interior or exterior wall."),
        _enum("load_bearing", Tristate, "Load bearing; unknown unless stated."),
    ),
    ElementKind.WALL_JOIN: (
        _typed("wall_a", "id", "", "Id of the first wall."),
        _typed("wall_b", "id", "", "Id of the second wall."),
        _enum("join_type", JoinType, "L corner or T join."),
        _enum("end_a", JoinEnd, "Part of wall A that takes part in the join."),
        _enum("end_b", JoinEnd, "Part of wall B that takes part in the join."),
    ),
    ElementKind.SLAB: (
        _typed("outline", "polygon", "m", "Closed outline, first point not repeated."),
        _num("thickness", "m", *_THICKNESS, "Slab thickness."),
        _num("top_elevation", "m", -100.0, 1000.0, 0.001, "Elevation of the top surface."),
        _typed("openings", "polygon_list", "m", "Closed opening outlines inside the slab."),
        _typed("layers", "layers", "m", "Layer set (material, thickness); sum equals thickness."),
    ),
    ElementKind.CEILING: (
        _typed("outline", "polygon", "m", "Closed outline, first point not repeated."),
        _num("elevation", "m", -100.0, 1000.0, 0.001, "Elevation of the ceiling surface."),
        _typed("plane_normal", "vector3", "", "Plane normal when not horizontal."),
        _num("plane_offset", "m", -10000.0, 10000.0, 0.001, "Plane offset along the normal."),
        _num("sag", "m", 0.0, 1.0, 0.001, "Maximum sag against the reference surface."),
        _num("thickness", "m", 0.005, 2.0, 0.005, "Ceiling thickness."),
        ParamSpec(
            "trim_to_walls",
            description="End the ceiling at the inner faces of the bounding walls.",
            value_type="boolean",
        ),
        _typed("region_seed", "point2", "m", "Point inside the piece taken when walls split it."),
    ),
    ElementKind.OPENING: (
        _typed("host_wall_id", "id", "", "Id of the host wall."),
        _num("position_along_axis", "m", 0.0, 1000.0, 0.005, "Distance from wall start to center."),
        _num("width", "m", 0.1, 10.0, 0.005, "Opening width."),
        _num("height", "m", 0.1, 10.0, 0.005, "Opening height."),
        _num("sill_height", "m", 0.0, 10.0, 0.005, "Sill height above the wall base."),
        _num("reveal_depth", "m", 0.0, 1.0, 0.005, "Depth of the reveal."),
        _enum("inside_side", Side, "Wall face that is the inside."),
    ),
    ElementKind.WINDOW: (
        _typed("opening_id", "id", "", "Id of the opening the window fills."),
        _enum("operation", WindowOperation, "Window operation type."),
        _num("frame_depth", "m", 0.01, 0.5, 0.001, "Frame depth if measured."),
        _num("sill_projection", "m", 0.0, 0.5, 0.005, "Sill projection beyond the wall face."),
        _num("sill_thickness", "m", 0.0, 0.3, 0.005, "Sill thickness."),
        _typed("mullion_positions", "number_list", "m", "Mullion bar centres from the left window edge."),
    ),
    ElementKind.DOOR: (
        _typed("opening_id", "id", "", "Id of the opening the door fills."),
        _enum("operation", DoorOperation, "Door operation type."),
        _enum("swing_side", Side, "Side of the door hinge."),
        _num("frame_depth", "m", 0.01, 0.5, 0.001, "Frame depth if measured."),
        _typed("glazed", "boolean", "", "The leaf carries a glass panel."),
    ),
    ElementKind.SPACE: (
        _typed("name", "string", "", "Room name."),
        _typed("number", "string", "", "Room number."),
        _typed("seed", "point2", "m", "Optional point (x, y) inside the room in the local frame."),
        _typed("outline", "polygon", "m", "Explicit room outline; replaces room derivation from walls."),
        _typed("holes", "polygon_list", "m", "Holes of the explicit room outline."),
    ),
    ElementKind.STAIR: (
        _typed("start", "point2", "m", "Plan point at the foot of the first riser, centred."),
        _num(
            "direction", "deg", -360.0, 360.0, 0.1, "Travel direction, counter-clockwise from +x."
        ),
        _num("base_offset", "m", -10.0, 10.0, 0.005, "Base elevation offset to the storey."),
        _num("width", "m", 0.3, 5.0, 0.005, "Stair width across the flight."),
        _num("step_count", "count", 2, 200, 1, "Total number of risers over all flights."),
        _num("riser_height", "m", 0.05, 0.30, 0.001, "Riser height."),
        _num("tread_length", "m", 0.10, 0.60, 0.001, "Tread length (going)."),
        _typed("flights", "flights", "", "Flights: riser count, landing length, landing outline."),
        _num("nosing", "m", 0.0, 0.1, 0.001, "Nosing projection of the treads."),
        _num("landing_thickness", "m", 0.05, 0.5, 0.005, "Landing slab thickness."),
    ),
}


def get_param_schema(kind: ElementKind | str) -> dict[str, Any]:
    """Return the JSON-serializable parameter schema of an element kind.

    Raises ``KeyError`` for kinds that have no schema yet (reserved kinds).
    """
    element_kind = ElementKind(kind)
    specs = PARAM_SCHEMAS[element_kind]
    return {"kind": element_kind.value, "params": [spec.to_dict() for spec in specs]}


def check_param_ranges(kind: ElementKind | str, params: Mapping[str, Any]) -> list[str]:
    """Return messages for numeric parameters outside their schema range."""
    messages: list[str] = []
    for spec in PARAM_SCHEMAS[ElementKind(kind)]:
        value = params.get(spec.name)
        if spec.value_type != "number" or value is None or isinstance(value, bool):
            continue
        if spec.minimum is not None and value < spec.minimum:
            messages.append(f"{spec.name}={value} is below the minimum {spec.minimum} {spec.unit}")
        if spec.maximum is not None and value > spec.maximum:
            messages.append(f"{spec.name}={value} is above the maximum {spec.maximum} {spec.unit}")
    return messages
