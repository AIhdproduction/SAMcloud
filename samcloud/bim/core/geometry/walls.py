"""Wall geometry: axis frame, justification faces, join cuts, footprint and quantities.

Conventions (see docs/GEOMETRY.md):

* Direction ``d`` is the unit vector from ``params.start`` to ``params.end``. The left normal is
  ``n = rot(+90 deg) d = (-dy, dx)``.
* The wall body occupies the signed offsets ``[lo, hi]`` from the axis along ``n``. The face at
  ``hi`` is the left face, the face at ``lo`` is the right face. Center: ``(-t/2, +t/2)``.
  Left: the axis is the left face, the body extends to the right, ``(-t, 0)``.
  Right: the axis is the right face, the body extends to the left, ``(0, +t)``.
* ``s`` is the distance along the axis measured from the untrimmed parameter start.
* Footprints are counter-clockwise, closed without repeating the first vertex, ordered
  ``(right face start, right face end, left face end, left face start)``.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from samcloud.bim.core.geometry._math import (
    Vec2,
    add_scaled,
    cross,
    dot,
    intersect_lines,
    left_normal,
    shifted,
    signed_area,
    sub,
    unshifted,
)
from samcloud.bim.core.model import (
    JoinEnd,
    JoinType,
    WallElement,
    WallJoinElement,
    WallJustification,
)

TOLERANCE = 1e-9
"""Geometric tolerance for containment checks (m)."""


@dataclass(frozen=True)
class OpeningCut:
    """Rectangular cut of an opening through a wall, in wall coordinates (clipped to the wall)."""

    opening_id: str
    s_min: float
    s_max: float
    z_min: float
    z_max: float
    area: float


@dataclass(frozen=True)
class WallGeometry:
    """Geometry and quantities of one wall. Coordinates are project coordinates (float64)."""

    element_id: str
    storey_id: str | None
    space_ids: tuple[str, ...]
    param_start: Vec2
    param_end: Vec2
    direction: Vec2
    normal: Vec2
    justification: WallJustification
    offset_right: float
    offset_left: float
    thickness: float
    s_start: float
    s_end: float
    axis_start: Vec2
    axis_end: Vec2
    length: float
    footprint: tuple[Vec2, ...]
    base_z: float
    top_z: float
    height: float
    gross_footprint_area: float
    gross_side_area: float
    net_side_area: float
    gross_volume: float
    net_volume: float
    opening_cuts: tuple[OpeningCut, ...] = ()
    issues: tuple[str, ...] = ()
    origin: Vec2 = (0.0, 0.0)
    """Local origin used for the computation."""
    local_footprint: tuple[Vec2, ...] = ()
    """Footprint relative to ``origin`` (exact, without the rounding of adding the origin)."""

    def footprint_relative_to(self, origin: Vec2) -> list[Vec2]:
        """Footprint shifted by ``origin`` (uses the exact local values when origins match)."""
        if origin == self.origin and self.local_footprint:
            return list(self.local_footprint)
        return shifted(self.footprint, origin)

    @property
    def untrimmed_length(self) -> float:
        """Length of the parameter axis ``start -> end``."""
        return math.hypot(
            self.param_end[0] - self.param_start[0], self.param_end[1] - self.param_start[1]
        )

    @property
    def quantities(self) -> dict[str, float]:
        """IFC4 Qto_WallBaseQuantities names."""
        return {
            "Length": self.length,
            "Width": self.thickness,
            "Height": self.height,
            "GrossSideArea": self.gross_side_area,
            "NetSideArea": self.net_side_area,
            "GrossVolume": self.gross_volume,
            "NetVolume": self.net_volume,
            "GrossFootprintArea": self.gross_footprint_area,
        }

    def point(self, s: float, offset: float = 0.0) -> Vec2:
        """Project point at axis distance ``s`` (from the untrimmed start) and normal offset."""
        return (
            self.param_start[0] + s * self.direction[0] + offset * self.normal[0],
            self.param_start[1] + s * self.direction[1] + offset * self.normal[1],
        )


def face_offsets(justification: WallJustification, thickness: float) -> tuple[float, float]:
    """Return ``(offset_right, offset_left)`` of the body relative to the axis."""
    if justification == WallJustification.LEFT:
        return (-thickness, 0.0)
    if justification == WallJustification.RIGHT:
        return (0.0, thickness)
    return (-thickness / 2.0, thickness / 2.0)


@dataclass(frozen=True)
class _Frame:
    """Wall frame in local (origin-shifted) coordinates."""

    start: Vec2
    direction: Vec2
    normal: Vec2
    length: float
    lo: float
    hi: float

    def face_point(self, offset: float) -> Vec2:
        return add_scaled(self.start, self.normal, offset)

    def out_direction(self, end: JoinEnd) -> Vec2:
        """Unit vector from the given end into the wall body."""
        if end == JoinEnd.START:
            return self.direction
        return (-self.direction[0], -self.direction[1])


def _frame(wall: WallElement, origin: Vec2) -> _Frame:
    params = wall.params
    start = (params.start[0] - origin[0], params.start[1] - origin[1])
    end = (params.end[0] - origin[0], params.end[1] - origin[1])
    length = math.hypot(end[0] - start[0], end[1] - start[1])
    direction = ((end[0] - start[0]) / length, (end[1] - start[1]) / length)
    lo, hi = face_offsets(params.justification, params.thickness)
    return _Frame(start, direction, left_normal(direction), length, lo, hi)


_EndPoints = tuple[Vec2, Vec2]
"""Cut vertices of a wall end: ``(point on the right face, point on the left face)``."""


def _inner_is_left(frame: _Frame, own_out: Vec2, other_out: Vec2) -> bool:
    """True when the left face of this wall is the face on the side of the other wall."""
    sign = 1.0 if cross(own_out, other_out) > 0.0 else -1.0
    side = left_normal(own_out)
    toward_other = (sign * side[0], sign * side[1])
    return dot(toward_other, frame.normal) > 0.0


def _l_cut(
    frame_a: _Frame, end_a: JoinEnd, frame_b: _Frame, end_b: JoinEnd
) -> tuple[_EndPoints, _EndPoints] | None:
    """Mitre cut of an L join (inner faces meet, outer faces meet); None if parallel."""
    out_a = frame_a.out_direction(end_a)
    out_b = frame_b.out_direction(end_b)
    if abs(cross(out_a, out_b)) < 1e-9:
        return None
    a_inner_left = _inner_is_left(frame_a, out_a, out_b)
    b_inner_left = _inner_is_left(frame_b, out_b, out_a)
    corner: dict[bool, Vec2] = {}
    for inner in (True, False):
        a_left = a_inner_left if inner else not a_inner_left
        b_left = b_inner_left if inner else not b_inner_left
        point = intersect_lines(
            frame_a.face_point(frame_a.hi if a_left else frame_a.lo),
            frame_a.direction,
            frame_b.face_point(frame_b.hi if b_left else frame_b.lo),
            frame_b.direction,
        )
        if point is None:
            return None
        corner[inner] = point
    inner_point, outer_point = corner[True], corner[False]
    a_cut = (outer_point, inner_point) if a_inner_left else (inner_point, outer_point)
    b_cut = (outer_point, inner_point) if b_inner_left else (inner_point, outer_point)
    return a_cut, b_cut


def _t_cut(frame_a: _Frame, end_a: JoinEnd, frame_b: _Frame) -> _EndPoints | None:
    """Cut wall A at the near face of wall B (the path wall); None if parallel."""
    out_a = frame_a.out_direction(end_a)
    near = frame_b.hi if dot(out_a, frame_b.normal) > 0.0 else frame_b.lo
    face_point = frame_b.face_point(near)
    right = intersect_lines(
        frame_a.face_point(frame_a.lo), frame_a.direction, face_point, frame_b.direction
    )
    left = intersect_lines(
        frame_a.face_point(frame_a.hi), frame_a.direction, face_point, frame_b.direction
    )
    if right is None or left is None:
        return None
    return right, left


def _axis_s(frame: _Frame, right: Vec2, left: Vec2) -> float:
    """Axis distance where the end cut line (through both face vertices) crosses the axis."""
    s_right = dot(sub(right, frame.start), frame.direction)
    s_left = dot(sub(left, frame.start), frame.direction)
    return s_right + (0.0 - frame.lo) * (s_left - s_right) / (frame.hi - frame.lo)


def compute_wall_geometries(
    walls: Sequence[WallElement],
    joins: Sequence[WallJoinElement] = (),
    storey_elevations: Mapping[str, float] | None = None,
    origin: Vec2 = (0.0, 0.0),
) -> tuple[dict[str, WallGeometry], dict[str, list[str]]]:
    """Compute wall geometry with join cuts.

    Returns ``(geometries by wall id, issues by join id)``. Issues of a wall itself are on its
    geometry. ``origin`` is a local origin subtracted internally for precision; results are in
    project coordinates. Joins to unknown walls, joins that do not fit their type, joins at an
    already cut wall end and parallel walls are reported and the affected end stays square.
    """
    elevations = storey_elevations or {}
    by_id = {wall.id: wall for wall in walls}
    frames = {wall.id: _frame(wall, origin) for wall in walls}
    cuts: dict[tuple[str, JoinEnd], _EndPoints] = {}
    join_issues: dict[str, list[str]] = {}

    def fail(join: WallJoinElement, message: str) -> None:
        join_issues.setdefault(join.id, []).append(message)

    for join in joins:
        p = join.params
        if p.wall_a not in by_id or p.wall_b not in by_id:
            fail(join, "join references an unknown wall")
            continue
        if p.join_type == JoinType.L:
            if JoinEnd.PATH in (p.end_a, p.end_b):
                fail(join, "L join needs start or end on both walls")
                continue
            keys = [(p.wall_a, p.end_a), (p.wall_b, p.end_b)]
        else:
            if (p.end_a, p.end_b).count(JoinEnd.PATH) != 1:
                fail(join, "T join needs exactly one path end")
                continue
            keys = [(p.wall_a, p.end_a)] if p.end_a != JoinEnd.PATH else [(p.wall_b, p.end_b)]
        taken = [key for key in keys if key in cuts]
        if taken:
            fail(join, f"wall end {taken[0][0]}/{taken[0][1].value} is already cut by another join")
            continue
        if p.join_type == JoinType.L:
            l_result = _l_cut(frames[p.wall_a], p.end_a, frames[p.wall_b], p.end_b)
            if l_result is None:
                fail(join, "joined walls are parallel; ends stay square")
                continue
            cuts[keys[0]] = l_result[0]
            cuts[keys[1]] = l_result[1]
        else:
            abutting, end, path = (
                (p.wall_a, p.end_a, p.wall_b)
                if p.end_a != JoinEnd.PATH
                else (p.wall_b, p.end_b, p.wall_a)
            )
            t_result = _t_cut(frames[abutting], end, frames[path])
            if t_result is None:
                fail(join, "T join walls are parallel; end stays square")
                continue
            cuts[(abutting, end)] = t_result

    result_map: dict[str, WallGeometry] = {}
    for wall in walls:
        frame = frames[wall.id]
        params = wall.params
        issues: list[str] = []
        start_cut = cuts.get((wall.id, JoinEnd.START))
        end_cut = cuts.get((wall.id, JoinEnd.END))
        if start_cut is None:
            start_cut = (frame.face_point(frame.lo), frame.face_point(frame.hi))
            s_start = 0.0
        else:
            s_start = _axis_s(frame, *start_cut)
        if end_cut is None:
            end_cut = (
                add_scaled(frame.face_point(frame.lo), frame.direction, frame.length),
                add_scaled(frame.face_point(frame.hi), frame.direction, frame.length),
            )
            s_end = frame.length
        else:
            s_end = _axis_s(frame, *end_cut)
        local_footprint = [start_cut[0], end_cut[0], end_cut[1], start_cut[1]]
        area = signed_area(local_footprint)
        length = s_end - s_start
        if length <= TOLERANCE or area <= 0.0:
            issues.append("wall has no positive extent after join cuts")
        storey = wall.storey_id
        if storey is not None and storey not in elevations:
            issues.append(f"unknown storey {storey!r}; elevation 0 used")
        elevation = elevations.get(storey, 0.0) if storey is not None else 0.0
        base_z = elevation + params.base_offset
        height = params.height
        axis_start = add_scaled(frame.start, frame.direction, s_start)
        axis_end = add_scaled(frame.start, frame.direction, s_end)
        gross_side = length * height
        gross_volume = area * height
        result_map[wall.id] = WallGeometry(
            element_id=wall.id,
            storey_id=storey,
            space_ids=tuple(wall.space_ids),
            param_start=(params.start[0], params.start[1]),
            param_end=(params.end[0], params.end[1]),
            direction=frame.direction,
            normal=frame.normal,
            justification=params.justification,
            offset_right=frame.lo,
            offset_left=frame.hi,
            thickness=params.thickness,
            s_start=s_start,
            s_end=s_end,
            axis_start=unshifted([axis_start], origin)[0],
            axis_end=unshifted([axis_end], origin)[0],
            length=length,
            footprint=unshifted(local_footprint, origin),
            base_z=base_z,
            top_z=base_z + height,
            height=height,
            gross_footprint_area=area,
            gross_side_area=gross_side,
            net_side_area=gross_side,
            gross_volume=gross_volume,
            net_volume=gross_volume,
            issues=tuple(issues),
            origin=origin,
            local_footprint=tuple(local_footprint),
        )
    return result_map, join_issues
