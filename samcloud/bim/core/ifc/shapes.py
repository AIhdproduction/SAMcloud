"""Small helpers that build IFC profiles, extrusions, representations and placements.

All lengths are SI metres (the file declares metre units). Profiles are 2D in the local XY plane
of the extrusion position.
"""

from __future__ import annotations

from collections.abc import Sequence

import ifcopenshell
import ifcopenshell.api.geometry
import numpy as np
import numpy.typing as npt

from samcloud.bim.core.ifc.context import ExportContext

Vec2 = tuple[float, float]
Vec3 = tuple[float, float, float]
Entity = ifcopenshell.entity_instance


def _point2(f: ifcopenshell.file, point: Vec2) -> Entity:
    return f.createIfcCartesianPoint((float(point[0]), float(point[1])))


def _point3(f: ifcopenshell.file, point: Vec3) -> Entity:
    return f.createIfcCartesianPoint((float(point[0]), float(point[1]), float(point[2])))


def _direction(f: ifcopenshell.file, vector: Vec3) -> Entity:
    return f.createIfcDirection((float(vector[0]), float(vector[1]), float(vector[2])))


def _closed_polyline(f: ifcopenshell.file, points: Sequence[Vec2]) -> Entity:
    ring = [_point2(f, p) for p in points]
    ring.append(ring[0])
    return f.createIfcPolyline(ring)


def rectangle_profile(f: ifcopenshell.file, center: Vec2, x_dim: float, y_dim: float) -> Entity:
    """Rectangle profile centred at ``center``."""
    placement = f.createIfcAxis2Placement2D(_point2(f, center), None)
    return f.createIfcRectangleProfileDef("AREA", None, placement, float(x_dim), float(y_dim))


def hollow_rectangle_profile(
    f: ifcopenshell.file, center: Vec2, x_dim: float, y_dim: float, wall_thickness: float
) -> Entity:
    """Rectangle frame profile (outer size x_dim by y_dim) centred at ``center``."""
    placement = f.createIfcAxis2Placement2D(_point2(f, center), None)
    return f.createIfcRectangleHollowProfileDef(
        "AREA", None, placement, float(x_dim), float(y_dim), float(wall_thickness), None, None
    )


def polygon_profile(
    f: ifcopenshell.file, outline: Sequence[Vec2], holes: Sequence[Sequence[Vec2]] = ()
) -> Entity:
    """Arbitrary closed profile; with holes an IfcArbitraryProfileDefWithVoids."""
    outer = _closed_polyline(f, outline)
    if holes:
        inner = [_closed_polyline(f, hole) for hole in holes]
        return f.createIfcArbitraryProfileDefWithVoids("AREA", None, outer, inner)
    return f.createIfcArbitraryClosedProfileDef("AREA", None, outer)


def extrusion(
    f: ifcopenshell.file,
    profile: Entity,
    depth: float,
    *,
    origin: Vec3 = (0.0, 0.0, 0.0),
    axis: Vec3 | None = None,
    ref_direction: Vec3 | None = None,
    direction: Vec3 = (0.0, 0.0, 1.0),
) -> Entity:
    """IfcExtrudedAreaSolid with its own position inside the representation."""
    position = f.createIfcAxis2Placement3D(
        _point3(f, origin),
        None if axis is None else _direction(f, axis),
        None if ref_direction is None else _direction(f, ref_direction),
    )
    return f.createIfcExtrudedAreaSolid(profile, position, _direction(f, direction), float(depth))


def swept_solid_representation(ctx: ExportContext, solids: Sequence[Entity]) -> Entity:
    """Body representation of one or more extruded solids (RepresentationType SweptSolid)."""
    return ctx.file.createIfcShapeRepresentation(ctx.body, "Body", "SweptSolid", list(solids))


def axis_representation(ctx: ExportContext, length: float) -> Entity:
    """Axis / Curve2D representation from (0, 0) to (length, 0) in the local frame."""
    representation = ifcopenshell.api.geometry.add_axis_representation(
        ctx.file, context=ctx.axis, axis=((0.0, 0.0), (float(length), 0.0))
    )
    assert representation is not None
    return representation


def assign_representations(
    ctx: ExportContext, product: Entity, representations: Sequence[Entity]
) -> None:
    """Attach representations to a product (creates its IfcProductDefinitionShape)."""
    for representation in representations:
        ifcopenshell.api.geometry.assign_representation(
            ctx.file, product=product, representation=representation
        )


def frame_matrix(origin: Vec3, x_axis: Vec3, z_axis: Vec3) -> npt.NDArray[np.float64]:
    """4x4 matrix with columns x, y = z cross x, z and the origin."""
    x = np.array(x_axis, dtype=np.float64)
    z = np.array(z_axis, dtype=np.float64)
    y = np.cross(z, x)
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, 0] = x
    matrix[:3, 1] = y
    matrix[:3, 2] = z
    matrix[:3, 3] = np.array(origin, dtype=np.float64)
    return matrix


def place(
    ctx: ExportContext,
    product: Entity,
    origin: Vec3,
    x_axis: Vec3 = (1.0, 0.0, 0.0),
    z_axis: Vec3 = (0.0, 0.0, 1.0),
) -> None:
    """Set the ObjectPlacement from a global frame (relative placement is derived by ifcopenshell).

    The product must already be contained in or aggregated to its parent so that the placement
    is chained to the parent placement.
    """
    ifcopenshell.api.geometry.edit_object_placement(
        ctx.file, product=product, matrix=frame_matrix(origin, x_axis, z_axis)
    )
