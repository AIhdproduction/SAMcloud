"""Opening, window and door geometry on a host wall.

Opening frame (see docs/GEOMETRY.md): the origin is the lower corner with the smallest ``u`` on
the LEFT face of the host wall. ``u`` runs along the wall direction, ``v`` is up (+z) and ``w``
runs from the left face into the wall (``w = u x v = -n``). The box spans ``u in [0, width]``,
``v in [0, height]``, ``w in [0, thickness]``.

``position_along_axis`` is the distance from the UNTRIMMED parameter start of the wall to the
opening center, so the value does not jump when joins change.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence
from dataclasses import dataclass

from shapely.geometry import box
from shapely.ops import unary_union

from samcloud.bim.core.geometry._math import Vec3
from samcloud.bim.core.geometry.walls import TOLERANCE, OpeningCut, WallGeometry
from samcloud.bim.core.model import (
    DoorElement,
    DoorOperation,
    OpeningElement,
    Side,
    WindowElement,
    WindowOperation,
)

CORNER_ORDER: tuple[tuple[int, int, int], ...] = (
    (0, 0, 0),
    (1, 0, 0),
    (1, 0, 1),
    (0, 0, 1),
    (0, 1, 0),
    (1, 1, 0),
    (1, 1, 1),
    (0, 1, 1),
)
"""Corner order of ``corners``: ``(iu, iv, iw)`` with 0 = minimum and 1 = maximum."""


@dataclass(frozen=True)
class OpeningGeometry:
    """Placed opening box in project coordinates."""

    element_id: str
    host_wall_id: str
    origin: Vec3
    u_axis: Vec3
    v_axis: Vec3
    w_axis: Vec3
    width: float
    height: float
    depth: float
    sill_height: float
    reveal_depth: float
    inside_side: Side
    s_min: float
    s_max: float
    z_min: float
    z_max: float
    corners: tuple[Vec3, ...]
    issues: tuple[str, ...] = ()

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def quantities(self) -> dict[str, float]:
        """Opening quantities (IFC4 Qto_OpeningElementBaseQuantities names) plus SillHeight."""
        return {
            "Width": self.width,
            "Height": self.height,
            "Area": self.area,
            "SillHeight": self.sill_height,
        }


@dataclass(frozen=True)
class FillingGeometry:
    """Window or door filling an opening. Overall size equals the opening size (no frame inset)."""

    element_id: str
    kind: str
    opening_id: str
    host_wall_id: str
    origin: Vec3
    corners: tuple[Vec3, ...]
    width: float
    height: float
    sill_height: float
    operation: WindowOperation | DoorOperation
    swing_side: Side | None
    issues: tuple[str, ...] = ()

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def quantities(self) -> dict[str, float]:
        """Window/door quantities (Width, Height, Area) plus SillHeight."""
        return {
            "Width": self.width,
            "Height": self.height,
            "Area": self.area,
            "SillHeight": self.sill_height,
        }


def compute_opening_geometry(opening: OpeningElement, wall: WallGeometry) -> OpeningGeometry:
    """Place an opening on its host wall and validate that it lies within the wall."""
    params = opening.params
    s_min = params.position_along_axis - params.width / 2.0
    s_max = params.position_along_axis + params.width / 2.0
    z_min = wall.base_z + params.sill_height
    z_max = z_min + params.height
    left = wall.offset_left
    origin_xy = wall.point(s_min, left)
    d, n = wall.direction, wall.normal
    u_axis: Vec3 = (d[0], d[1], 0.0)
    v_axis: Vec3 = (0.0, 0.0, 1.0)
    w_axis: Vec3 = (-n[0], -n[1], 0.0)
    corners = tuple(
        (
            origin_xy[0] + iu * params.width * u_axis[0] + iw * wall.thickness * w_axis[0],
            origin_xy[1] + iu * params.width * u_axis[1] + iw * wall.thickness * w_axis[1],
            z_min + iv * params.height,
        )
        for iu, iv, iw in CORNER_ORDER
    )
    issues: list[str] = []
    if s_min < -TOLERANCE:
        issues.append("opening starts before the wall start")
    if s_max > wall.untrimmed_length + TOLERANCE:
        issues.append("opening ends after the wall end")
    if not issues and (s_min < wall.s_start - TOLERANCE or s_max > wall.s_end + TOLERANCE):
        issues.append("opening lies in a trimmed part of the wall (join cut)")
    if z_max > wall.top_z + TOLERANCE:
        issues.append("opening extends above the wall top")
    return OpeningGeometry(
        element_id=opening.id,
        host_wall_id=params.host_wall_id,
        origin=(origin_xy[0], origin_xy[1], z_min),
        u_axis=u_axis,
        v_axis=v_axis,
        w_axis=w_axis,
        width=params.width,
        height=params.height,
        depth=wall.thickness,
        sill_height=params.sill_height,
        reveal_depth=params.reveal_depth,
        inside_side=params.inside_side,
        s_min=s_min,
        s_max=s_max,
        z_min=z_min,
        z_max=z_max,
        corners=corners,
        issues=tuple(issues),
    )


def compute_filling_geometry(
    element: WindowElement | DoorElement, opening: OpeningGeometry
) -> FillingGeometry:
    """Window or door geometry: the opening box; operation and swing are passed through."""
    is_door = isinstance(element, DoorElement)
    return FillingGeometry(
        element_id=element.id,
        kind="door" if is_door else "window",
        opening_id=opening.element_id,
        host_wall_id=opening.host_wall_id,
        origin=opening.origin,
        corners=opening.corners,
        width=opening.width,
        height=opening.height,
        sill_height=opening.sill_height,
        operation=element.params.operation,
        swing_side=element.params.swing_side if isinstance(element, DoorElement) else None,
        issues=(),
    )


def apply_openings(wall: WallGeometry, openings: Sequence[OpeningGeometry]) -> WallGeometry:
    """Return the wall with its opening cuts, net side area and net volume.

    Each opening is clipped to the wall side face (trimmed axis range, base to top). Overlapping
    openings are counted once (union). The cut goes through the full thickness.
    """
    cuts: list[OpeningCut] = []
    rectangles = []
    for opening in openings:
        s_min = max(opening.s_min, wall.s_start)
        s_max = min(opening.s_max, wall.s_end)
        z_min = max(opening.z_min, wall.base_z)
        z_max = min(opening.z_max, wall.top_z)
        if s_max <= s_min or z_max <= z_min:
            continue
        rectangle = box(s_min, z_min - wall.base_z, s_max, z_max - wall.base_z)
        rectangles.append(rectangle)
        cuts.append(OpeningCut(opening.element_id, s_min, s_max, z_min, z_max, rectangle.area))
    cut_area = float(unary_union(rectangles).area) if rectangles else 0.0
    return dataclasses.replace(
        wall,
        opening_cuts=tuple(cuts),
        net_side_area=wall.gross_side_area - cut_area,
        net_volume=wall.gross_volume - cut_area * wall.thickness,
    )
