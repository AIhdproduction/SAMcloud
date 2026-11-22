"""IfcSpace with body and first level IfcRelSpaceBoundary to walls, floors and ceilings.

Space body: the room polygon (with holes) extruded from the floor top to the ceiling underside
(or the storey height, see docs/GEOMETRY.md). The placement origin is the lower left corner of the
polygon bounding box at floor level, axes parallel to the model axes.

Space boundaries are plain ``IfcRelSpaceBoundary`` (first level) with an
``IfcConnectionSurfaceGeometry`` (IfcCurveBoundedPlane) whose placement is relative to the SPACE
placement. The plane normal points away from the space (into the bounding element); this
convention is recorded as an open point in docs/IFC_MAPPING.md (R5). Walls: the face on the room
side (the face with the larger shared length with the room boundary) over the shared interval and
the clear height. Floors: the room polygon at the floor top (normal down). Ceilings: the room
polygon at the ceiling underside (normal up, horizontal also for tilted ceilings).
"""

from __future__ import annotations

import math

import ifcopenshell
import ifcopenshell.api.aggregate
import ifcopenshell.api.boundary
import ifcopenshell.api.root
import ifcopenshell.guid
import ifcopenshell.util.unit
from shapely.geometry import LineString
from shapely.geometry.base import BaseGeometry

from samcloud.bim.core.geometry import SpaceGeometry, WallGeometry
from samcloud.bim.core.ifc import identity, shapes
from samcloud.bim.core.ifc.context import ExportContext
from samcloud.bim.core.ifc.ids import set_guid
from samcloud.bim.core.ifc.spatial import container_of
from samcloud.bim.core.model import SpaceElement, WallElement, WallFunction

Entity = ifcopenshell.entity_instance

TOUCH_TOLERANCE = 1e-3
"""Distance between a wall face and the room boundary that still counts as shared (m)."""
MIN_BOUNDARY_LENGTH = 1e-6


def write_spaces(ctx: ExportContext) -> None:
    """IfcSpace per scene space: Name = number, LongName = name, body from the room polygon."""
    f = ctx.file
    for element in ctx.scene.elements.values():
        if not isinstance(element, SpaceElement):
            continue
        number = element.params.number
        name = element.params.name or element.name
        space = ifcopenshell.api.root.create_entity(
            f, ifc_class="IfcSpace", predefined_type="INTERNAL", name=number or name or "Space"
        )
        space.LongName = name or None
        space.Description = element.description or identity.space_description(number, name)
        space.ObjectType = identity.SPACE_OBJECT_TYPE
        space.CompositionType = "ELEMENT"
        set_guid(ctx, space, element.id)
        ctx.entities[element.id] = space
        if element.storey_id is None:
            ctx.flag(element.id, "no storey assigned (aggregated to the building)")
        container = container_of(ctx, element.storey_id, element.building_id)
        ifcopenshell.api.aggregate.assign_object(f, products=[space], relating_object=container)

        geometry = ctx.geometry.spaces.get(element.id)
        if geometry is None:
            issues = ctx.geometry.issues.get(element.id, ())
            ctx.flag(
                element.id,
                "no room polygon found: space exported without body (" + "; ".join(issues) + ")",
            )
            continue
        origin = (
            min(p[0] for p in geometry.exterior),
            min(p[1] for p in geometry.exterior),
            geometry.floor_z,
        )
        ctx.space_frames[element.id] = origin
        shapes.place(ctx, space, origin)
        if geometry.height <= 0.0:
            ctx.flag(element.id, "space height is not positive: space exported without body")
            continue
        ring = [(x - origin[0], y - origin[1]) for x, y in geometry.exterior]
        holes = [[(x - origin[0], y - origin[1]) for x, y in hole] for hole in geometry.holes]
        profile = shapes.polygon_profile(f, ring, holes)
        solid = shapes.extrusion(f, profile, geometry.height)
        shapes.assign_representations(ctx, space, [shapes.swept_solid_representation(ctx, [solid])])


def _boundary(
    ctx: ExportContext,
    space: Entity,
    element: Entity,
    internal_external: str,
    description: str = "1stLevel",
) -> Entity:
    return ctx.file.create_entity(
        "IfcRelSpaceBoundary",
        GlobalId=ifcopenshell.guid.new(),
        Description=description,
        RelatingSpace=space,
        RelatedBuildingElement=element,
        PhysicalOrVirtualBoundary="PHYSICAL",
        InternalOrExternalBoundary=internal_external,
    )


def _wall_face(
    wall: WallGeometry, room_boundary: BaseGeometry
) -> tuple[tuple[float, float], tuple[float, float], bool, float, float] | None:
    """Room side face of a wall: (face start, direction, is_left_face, t0, t1) or None."""
    footprint = wall.footprint
    faces = {
        False: (footprint[0], footprint[1]),
        True: (footprint[3], footprint[2]),
    }
    best: tuple[float, bool] | None = None
    shared_by_face: dict[bool, object] = {}
    for is_left, (start, end) in faces.items():
        line = LineString([start, end])
        shared = room_boundary.intersection(line.buffer(TOUCH_TOLERANCE))
        shared_by_face[is_left] = shared
        length = float(shared.length)
        if length > MIN_BOUNDARY_LENGTH and (best is None or length > best[0]):
            best = (length, is_left)
    if best is None:
        return None
    is_left = best[1]
    start, end = faces[is_left]
    face_length = math.hypot(end[0] - start[0], end[1] - start[1])
    if face_length <= MIN_BOUNDARY_LENGTH:
        return None
    direction = ((end[0] - start[0]) / face_length, (end[1] - start[1]) / face_length)
    shared = shared_by_face[is_left]
    coordinates: list[tuple[float, float]] = []
    for part in getattr(shared, "geoms", [shared]):
        coordinates.extend((float(x), float(y)) for x, y in part.coords)
    ts = [(x - start[0]) * direction[0] + (y - start[1]) * direction[1] for x, y in coordinates]
    t0 = max(0.0, min(ts))
    t1 = min(face_length, max(ts))
    if t1 - t0 <= MIN_BOUNDARY_LENGTH:
        return None
    return start, direction, is_left, t0, t1


def _wall_boundary(
    ctx: ExportContext,
    space_entity: Entity,
    space_geometry: SpaceGeometry,
    space_origin: tuple[float, float, float],
    wall_element: WallElement,
    room_boundary: BaseGeometry,
) -> None:
    wall_geometry = ctx.geometry.walls.get(wall_element.id)
    wall_entity = ctx.entities.get(wall_element.id)
    if wall_geometry is None or wall_entity is None:
        return
    external = "EXTERNAL" if wall_element.params.function == WallFunction.EXTERIOR else "INTERNAL"
    relation = _boundary(ctx, space_entity, wall_entity, external)
    face = _wall_face(wall_geometry, room_boundary)
    if face is None:
        ctx.warn(
            f"space {space_geometry.element_id}: wall {wall_element.id} has no shared face; "
            "space boundary written without connection geometry"
        )
        return
    start, direction, is_left, t0, t1 = face
    normal = wall_geometry.normal
    # Normal points away from the space, into the wall body.
    into = (-normal[0], -normal[1]) if is_left else normal
    ref = (-into[1], into[0])
    z0 = max(space_geometry.floor_z, wall_geometry.base_z)
    z1 = min(space_geometry.ceiling_z, wall_geometry.top_z)
    if z1 <= z0:
        z0, z1 = wall_geometry.base_z, wall_geometry.top_z
    p0 = (start[0] + t0 * direction[0], start[1] + t0 * direction[1])
    p1 = (start[0] + t1 * direction[0], start[1] + t1 * direction[1])
    anchor = p0 if p0[0] * ref[0] + p0[1] * ref[1] <= p1[0] * ref[0] + p1[1] * ref[1] else p1
    length = t1 - t0
    ifcopenshell.api.boundary.assign_connection_geometry(
        ctx.file,
        rel_space_boundary=relation,
        outer_boundary=[(0.0, 0.0), (length, 0.0), (length, z1 - z0), (0.0, z1 - z0)],
        location=(anchor[0] - space_origin[0], anchor[1] - space_origin[1], z0 - space_origin[2]),
        axis=(into[0], into[1], 0.0),
        ref_direction=(ref[0], ref[1], 0.0),
    )


def _horizontal_boundary(
    ctx: ExportContext,
    space_entity: Entity,
    space_geometry: SpaceGeometry,
    space_origin: tuple[float, float, float],
    element: Entity,
    z: float,
    downward: bool,
) -> None:
    relation = _boundary(ctx, space_entity, element, "NOTDEFINED")
    # Local Y = Z x X: with Z down (floor) local y is the negative global y.
    sign = -1.0 if downward else 1.0

    def ring(points: tuple[tuple[float, float], ...]) -> list[tuple[float, float]]:
        return [(x - space_origin[0], sign * (y - space_origin[1])) for x, y in points]

    ifcopenshell.api.boundary.assign_connection_geometry(
        ctx.file,
        rel_space_boundary=relation,
        outer_boundary=ring(space_geometry.exterior),
        location=(0.0, 0.0, z - space_origin[2]),
        axis=(0.0, 0.0, -1.0 if downward else 1.0),
        ref_direction=(1.0, 0.0, 0.0),
    )
    if space_geometry.holes:
        # (assign_connection_geometry stacks all inner boundaries into one array and fails on holes with different vertex counts)
        scale = ifcopenshell.util.unit.calculate_unit_scale(ctx.file)
        inner = []
        for hole in space_geometry.holes:
            points = [ctx.file.createIfcCartesianPoint((float(x) / scale, float(y) / scale)) for x, y in ring(hole)]
            inner.append(ctx.file.createIfcPolyline([*points, points[0]]))
        relation.ConnectionGeometry.SurfaceOnRelatingElement.InnerBoundaries = inner


def write_space_boundaries(ctx: ExportContext) -> None:
    """First level IfcRelSpaceBoundary from every space to its walls, floors and ceilings."""
    for element in ctx.scene.elements.values():
        if not isinstance(element, SpaceElement):
            continue
        geometry = ctx.geometry.spaces.get(element.id)
        space = ctx.entities.get(element.id)
        origin = ctx.space_frames.get(element.id)
        if geometry is None or space is None or origin is None:
            continue
        room_boundary = geometry.polygon().boundary
        for wall_id in geometry.bounding_wall_ids:
            wall_element = ctx.scene.elements.get(wall_id)
            if isinstance(wall_element, WallElement):
                _wall_boundary(ctx, space, geometry, origin, wall_element, room_boundary)
        for floor_id in geometry.floor_ids:
            floor = ctx.entities.get(floor_id)
            if floor is not None:
                _horizontal_boundary(ctx, space, geometry, origin, floor, geometry.floor_z, True)
        for ceiling_id in geometry.ceiling_ids:
            ceiling = ctx.entities.get(ceiling_id)
            if ceiling is not None:
                _horizontal_boundary(
                    ctx, space, geometry, origin, ceiling, geometry.ceiling_z, False
                )
