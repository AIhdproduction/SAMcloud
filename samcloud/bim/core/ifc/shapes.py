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


def triangulated_face_set(
    ifc: ifcopenshell.file,
    vertices: Sequence[Vec3],
    triangles: Sequence[tuple[int, int, int]],
) -> Entity:
    """Create a closed IFC triangular face set from zero-based mesh indices."""
    points = ifc.createIfcCartesianPointList3D(
        [tuple(float(value) for value in vertex) for vertex in vertices]
    )
    indices = [tuple(int(index) + 1 for index in triangle) for triangle in triangles]
    return ifc.createIfcTriangulatedFaceSet(points, None, True, indices, None)


def tessellation_representation(ctx: ExportContext, item: Entity | Sequence[Entity]) -> Entity:
    """Body/Tessellation representation for one triangulated mesh item or several items in ONE representation."""
    items = list(item) if isinstance(item, (list, tuple)) else [item]
    return ctx.file.createIfcShapeRepresentation(ctx.body, "Body", "Tessellation", items)


def swept_curb(
    ifc: ifcopenshell.file,
    axis: Sequence[Vec3],
    width: float = 0.12,
    height: float = 0.25,
    *,
    down: bool = False,
) -> Entity:
    """Create an IfcFixedReferenceSweptAreaSolid along a three-dimensional polyline.

    The curb stands on the directrix by default; with ``down`` it hangs below it (directrix = top edge).
    """
    profile = rectangle_profile(ifc, (0.0, height / 2.0 if down else -height / 2.0), width, height)
    directrix = ifc.createIfcPolyline([_point3(ifc, point) for point in axis])
    # Directrix uses absolute coordinates, so the solid position stays at the origin
    # (an axis[0] offset here would be applied a second time). IfcOpenShell maps the
    # profile +Y to -Z for FixedReference (0, 0, 1), so the profile hangs at -height/2
    # to make the curb stand up from the axis.
    position = ifc.createIfcAxis2Placement3D(_point3(ifc, (0.0, 0.0, 0.0)), None, None)
    return ifc.createIfcFixedReferenceSweptAreaSolid(
        profile, position, directrix, None, None, _direction(ifc, (0.0, 0.0, 1.0))
    )


_CROWN_SECTORS = 12
# Unit profiles (height fraction h from the crown base, radius fraction r) of the lathe crown meshes.
# Radius 0 at an end closes it with a single pole vertex, otherwise with a centre-fan cap.
_CROWN_PROFILES: dict[str, list[tuple[float, float]]] = {
    "ellipsoid": [(0.0, 0.0), (0.25, float(np.cos(np.pi / 6.0))), (0.5, 1.0), (0.75, float(np.cos(np.pi / 6.0))), (1.0, 0.0)],
    "cone": [(0.0, 1.0), (1.0 / 3.0, 2.0 / 3.0), (2.0 / 3.0, 1.0 / 3.0), (1.0, 0.0)],
    "umbrella": [(0.0, 0.6), (0.4, 1.0), (0.75, 0.85), (1.0, 0.0)],   # flattened, rounded dome
    "cylinder": [(0.0, 1.0), (0.5, 1.0), (1.0, 1.0)],
}
_CROWN_PROFILES["sphere"] = _CROWN_PROFILES["ellipsoid"]


def crown_mesh(
    centre_xy: Vec2, base_z: float, height: float, diameter: float, shape: str = "ellipsoid",
) -> tuple[list[Vec3], list[tuple[int, int, int]]]:
    """Closed lathe mesh (72 triangles) of a crown whose lowest point is exactly ``base_z``.

    ``shape`` is sphere/ellipsoid, cone (apex up), umbrella (flat dome) or cylinder (72 triangles each,
    so that crown + trunk mesh stay within the 60..120 triangle budget); unknown names fall back to the ellipsoid.
    """
    profile = _CROWN_PROFILES.get(shape, _CROWN_PROFILES["ellipsoid"])
    radius = diameter / 2.0
    cx, cy = centre_xy
    vertices: list[Vec3] = []
    rings: list[int | None] = []      # first vertex index of each full ring, None for a pole
    pole_index: dict[int, int] = {}
    for position, (h, rr) in enumerate(profile):
        z = base_z + height * h
        if rr <= 1e-9:
            pole_index[position] = len(vertices)
            vertices.append((cx, cy, z))
            rings.append(None)
            continue
        rings.append(len(vertices))
        for sector in range(_CROWN_SECTORS):
            azimuth = 2.0 * np.pi * sector / _CROWN_SECTORS
            vertices.append((cx + radius * rr * np.cos(azimuth), cy + radius * rr * np.sin(azimuth), z))
    triangles: list[tuple[int, int, int]] = []
    last = len(profile) - 1
    # Bottom end: pole fan, or a centre vertex closing the first ring.
    if rings[0] is None:
        bottom_pole, first = pole_index[0], rings[1]
    else:
        bottom_pole, first = len(vertices), rings[0]
        vertices.append((cx, cy, base_z))
    for sector in range(_CROWN_SECTORS):
        triangles.append((bottom_pole, first + (sector + 1) % _CROWN_SECTORS, first + sector))
    for position in range(last):
        lower, upper = rings[position], rings[position + 1]
        if lower is None or upper is None:
            continue
        for sector in range(_CROWN_SECTORS):
            nxt = (sector + 1) % _CROWN_SECTORS
            triangles.append((lower + sector, lower + nxt, upper + nxt))
            triangles.append((lower + sector, upper + nxt, upper + sector))
    if rings[last] is None:
        top_pole, ring = pole_index[last], rings[last - 1]
    else:
        top_pole, ring = len(vertices), rings[last]
        vertices.append((cx, cy, base_z + height))
    for sector in range(_CROWN_SECTORS):
        triangles.append((ring + sector, ring + (sector + 1) % _CROWN_SECTORS, top_pole))
    return vertices, triangles


def trunk_mesh(
    centre_xy: Vec2, base_z: float, height: float, diameter: float, sides: int = 12,
) -> tuple[list[Vec3], list[tuple[int, int, int]]]:
    """Closed triangulated cylinder (``sides`` sides, 2*sides + 2*(sides-2) triangles; 44 for 12 sides)."""
    cx, cy = centre_xy
    radius = diameter / 2.0
    vertices: list[Vec3] = []
    for z in (base_z, base_z + height):
        for k in range(sides):
            az = 2.0 * np.pi * k / sides
            vertices.append((cx + radius * np.cos(az), cy + radius * np.sin(az), z))
    triangles: list[tuple[int, int, int]] = []
    for k in range(sides):
        n = (k + 1) % sides
        triangles.append((k, n, sides + n))
        triangles.append((k, sides + n, sides + k))
    for k in range(1, sides - 1):
        triangles.append((0, k + 1, k))                                  # bottom fan (faces down)
        triangles.append((sides, sides + k, sides + k + 1))              # top fan (faces up)
    return vertices, triangles


_MIN_CROWN_CLEARANCE_M = 0.3     # crown base at least this far above the terrain
_TRUNK_INTO_CROWN = 0.15         # the trunk reaches this fraction of the crown height into the crown


def stylised_tree(
    ifc: ifcopenshell.file,
    trunk_xy: Vec2,
    base_z: float,
    height: float,
    crown_diameter: float,
    trunk_diameter: float,
    crown_shape: str = "ellipsoid",
    crown_base_z: float | None = None,
    crown_offset: Vec2 = (0.0, 0.0),
) -> tuple[Entity, Entity]:
    """Return ``(trunk, crown)`` as two closed IfcTriangulatedFaceSet items for ONE Body/Tessellation representation.

    The trunk is a 12-sided cylinder from the terrain (``base_z``) to the crown base; the crown (72 triangles)
    is centred on ``trunk_xy + crown_offset`` (the trunk may stand off-centre under the crown).  Its lowest point is ``crown_base_z`` (default: the
    legacy ``height - max(0.55 * height, 0.6 * crown_diameter)`` rule, never below ``base_z + 0.3``); the trunk
    runs up to ``crown_base + 0.15 * crown height`` so that both meshes overlap and look connected.
    """
    if crown_base_z is None:
        crown_height = max(height * 0.55, crown_diameter * 0.6)
        crown_base = base_z + height - crown_height
    else:
        crown_base = float(crown_base_z)
        crown_height = base_z + height - crown_base
        if crown_height < 0.2 * height or crown_height <= 0.0:
            crown_height = max(height * 0.55, crown_diameter * 0.6)
            crown_base = base_z + height - crown_height
    if crown_base < base_z + _MIN_CROWN_CLEARANCE_M:
        crown_base = base_z + _MIN_CROWN_CLEARANCE_M
        crown_height = max(base_z + height - crown_base, 0.2 * height)
    tv, tt = trunk_mesh(trunk_xy, base_z, crown_base + _TRUNK_INTO_CROWN * crown_height - base_z, trunk_diameter)
    centre = (trunk_xy[0] + crown_offset[0], trunk_xy[1] + crown_offset[1])
    cv, ct = crown_mesh(centre, crown_base, crown_height, crown_diameter, crown_shape)
    assert len(ct) <= 120 and 60 <= len(ct) + len(tt) <= 120
    return triangulated_face_set(ifc, tv, tt), triangulated_face_set(ifc, cv, ct)


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


# --------------------------------------------------------------------------------------
# presentation styles and layers (outdoor category styling)
# --------------------------------------------------------------------------------------

def _colour(f: ifcopenshell.file, rgb: Sequence[int]) -> Entity:
    return f.createIfcColourRgb(None, *(float(value) / 255.0 for value in rgb))


def surface_style(f: ifcopenshell.file, name: str, rgb: Sequence[int], transparency: float = 0.0) -> Entity:
    """IfcSurfaceStyle with one IfcSurfaceStyleRendering in the given 0..255 colour."""
    rendering = f.createIfcSurfaceStyleRendering(
        _colour(f, rgb), float(transparency), None, None, None, None, None, None, "NOTDEFINED"
    )
    return f.createIfcSurfaceStyle(name, "BOTH", [rendering])


def curve_style(f: ifcopenshell.file, name: str, rgb: Sequence[int]) -> Entity:
    """IfcCurveStyle (solid line, colour only) for curve items such as markings and footprints."""
    return f.createIfcCurveStyle(name, None, None, _colour(f, rgb), None)


def style_item(f: ifcopenshell.file, item: Entity, style: Entity) -> None:
    """Attach ``style`` to a representation item; an existing IfcStyledItem is re-pointed, never doubled."""
    existing = list(getattr(item, "StyledByItem", None) or [])
    if existing:
        existing[0].Styles = [style]
        return
    f.createIfcStyledItem(item, [style], None)


def presentation_layer(f: ifcopenshell.file, layers: dict[str, Entity], name: str) -> Entity:
    """Return the IfcPresentationLayerAssignment called ``name``, creating it on first use (cached in ``layers``)."""
    layer = layers.get(name)
    if layer is None:
        layer = f.createIfcPresentationLayerAssignment(name, None, [], None)
        layers[name] = layer
    return layer


def add_to_layer(layer: Entity, representation: Entity) -> None:
    """Add a shape representation to a presentation layer once."""
    assigned = list(layer.AssignedItems)
    if representation not in assigned:
        layer.AssignedItems = assigned + [representation]
