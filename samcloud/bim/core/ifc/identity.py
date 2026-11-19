"""Generated Description texts for exported elements (MASTERPROMPT 10.1 point 2).

Every exported element carries a Description: the scene element's own ``description`` when set,
otherwise a deterministic English text with the key facts (metres, two decimals). ObjectType is the
name of the type object when the element has one, otherwise the fixed texts below.
"""

from __future__ import annotations

OPENING_OBJECT_TYPE = "Opening"
SPACE_OBJECT_TYPE = "Space"


def metres(value: float) -> str:
    """Length in metres with two decimals."""
    return f"{value:.2f}"


def wall_description(function: str, thickness: float, height: float) -> str:
    """E.g. ``Exterior wall, 0.30 m thick, 2.80 m high``."""
    kind = {"exterior": "Exterior wall", "interior": "Interior wall"}.get(function, "Wall")
    return f"{kind}, {metres(thickness)} m thick, {metres(height)} m high"


def slab_description(predefined: str, thickness: float) -> str:
    """E.g. ``Floor slab, 0.25 m``."""
    kind = {"FLOOR": "Floor slab", "BASESLAB": "Base slab"}.get(predefined, "Slab")
    return f"{kind}, {metres(thickness)} m"


def ceiling_description(thickness: float) -> str:
    """E.g. ``Ceiling, 0.10 m``."""
    return f"Ceiling, {metres(thickness)} m"


def opening_description(width: float, height: float, wall_name: str) -> str:
    """E.g. ``Opening 1.20 x 1.40 m in Wall 1``."""
    return f"Opening {metres(width)} x {metres(height)} m in {wall_name}"


def window_description(width: float, height: float, sill_height: float) -> str:
    """E.g. ``Window 1.20 x 1.40 m, sill 0.90 m``."""
    return f"Window {metres(width)} x {metres(height)} m, sill {metres(sill_height)} m"


def door_description(width: float, height: float, operation: str) -> str:
    """E.g. ``Door 0.90 x 2.10 m, single swing left``."""
    return f"Door {metres(width)} x {metres(height)} m, {operation.replace('_', ' ')}"


def stair_description(
    predefined: str, risers: int, riser_height: float, tread_length: float, width: float
) -> str:
    """E.g. ``Straight run stair, 14 risers of 0.18 m, tread 0.28 m, 1.00 m wide``."""
    kind = {
        "STRAIGHT_RUN_STAIR": "Straight run stair",
        "TWO_STRAIGHT_RUN_STAIR": "Two straight run stair",
    }.get(predefined, "Stair")
    return (
        f"{kind}, {risers} risers of {metres(riser_height)} m, tread {metres(tread_length)} m, "
        f"{metres(width)} m wide"
    )


def space_description(number: str, name: str) -> str:
    """E.g. ``Room 001 Room 1`` (number and name, either may be empty)."""
    return " ".join(part for part in ("Room", number, name) if part)
