"""Walls, wall joins, slabs, ceilings, openings, windows and doors.

All geometry comes from :func:`samcloud.bim.core.geometry.compute_scene_geometry`; this module only maps it
to IFC entities, placements and representations.

Decisions (see docs/IFC_MAPPING.md, "Implementation notes (WP1.5)"):

* Wall body: the trimmed footprint extruded over the wall height. A footprint that is an exact
  rectangle in the wall frame (no join cuts, tolerance 1 micrometre) is written as an
  IfcRectangleProfileDef; mitred or trimmed footprints as an IfcArbitraryClosedProfileDef. The
  wall placement origin is the trimmed axis start, local X runs along the axis, local Y is the
  left normal, and the layer set usage offset equals the lower body offset (right face).
* Floor slab: profile with holes extruded in the NEGATIVE direction from a placement at the top
  face. Ceiling: placement at the underside, extruded upward. Tilted ceilings use the plane: the
  profile lies in the tilted plane (per-vertex z from the geometry) and the extrusion direction
  stays vertical, so the vertical thickness equals the parameter.
* Openings: box of the opening geometry, depth equal to the wall thickness, placed in the frame
  (x along the wall, y = left normal, z up) with its lower left corner on the left wall face.
* Windows and doors: per-occurrence geometry (frame and panel as extruded solids), no
  RepresentationMaps on the type. The overall size equals the opening size.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import ifcopenshell
import ifcopenshell.api.aggregate
import ifcopenshell.api.feature
import ifcopenshell.api.geometry
import ifcopenshell.api.material
import ifcopenshell.api.root
import ifcopenshell.api.spatial
import ifcopenshell.api.style
import ifcopenshell.api.type
import ifcopenshell.util.element

from samcloud.bim.core.geometry import (
    CeilingGeometry,
    FillingGeometry,
    OpeningGeometry,
    SlabGeometry,
    WallGeometry,
)
from samcloud.bim.core.ifc import identity, shapes
from samcloud.bim.core.ifc.context import ExportContext, is_a
from samcloud.bim.core.ifc.ids import set_guid
from samcloud.bim.core.ifc.spatial import container_of
from samcloud.bim.core.ifc.types import (
    DEFAULT_FRAME_DEPTH,
    LINING_THICKNESS,
    MULLION_WIDTH,
    PANEL_DEPTH,
    door_operation,
    door_type,
    layered_type,
    window_pane_count,
    window_partitioning,
    window_type,
)
from samcloud.bim.core.model import (
    CeilingElement,
    CeilingIfcClass,
    DoorElement,
    Element,
    ElementKind,
    JoinEnd,
    MaterialLayer,
    OpeningElement,
    SlabElement,
    WallElement,
    WallJoinElement,
    WindowElement,
)

Entity = ifcopenshell.entity_instance

RECTANGLE_TOLERANCE = 1e-6
"""A wall footprint within this distance of a rectangle is written as a rectangle profile (m)."""
GLAZING_THICKNESS = 0.02

JOIN_END: dict[JoinEnd, str] = {
    JoinEnd.START: "ATSTART",
    JoinEnd.END: "ATEND",
    JoinEnd.PATH: "ATPATH",
}

_ALLOWED_CLASSES: dict[ElementKind, tuple[str, ...]] = {
    ElementKind.WALL: ("IfcWall", "IfcWallStandardCase"),
    ElementKind.SLAB: ("IfcSlab",),
    ElementKind.OPENING: ("IfcOpeningElement",),
    ElementKind.WINDOW: ("IfcWindow",),
    ElementKind.DOOR: ("IfcDoor",),
}


def element_name(element: Element, label: str) -> str:
    """IFC Name: the scene name, else a label with the tag or the short id."""
    return element.name or f"{label} {element.tag or element.id[:8]}"


def enum_items(ctx: ExportContext, enum_name: str) -> tuple[str, ...]:
    """Enumeration literals of the IFC schema of the file."""
    schema = ifcopenshell.schema_by_name(ctx.file.schema)
    enumeration = schema.declaration_by_name(enum_name).as_enumeration_type()
    assert enumeration is not None, enum_name
    return tuple(enumeration.enumeration_items())


def valid_predefined(ctx: ExportContext, element: Element, enum_name: str, fallback: str) -> str:
    """The element's predefined type when it is a valid enumeration literal, else the fallback."""
    value = element.predefined_type
    if value is None:
        return fallback
    if value in enum_items(ctx, enum_name):
        return value
    ctx.flag(element.id, f"invalid PredefinedType {value!r} for {enum_name}; {fallback} used")
    return fallback


def ifc_class_for(ctx: ExportContext, element: Element, default: str) -> str:
    """The element's ifc_class when allowed for its kind, else the default."""
    allowed = _ALLOWED_CLASSES.get(element.kind, (default,))
    if element.ifc_class in allowed:
        return element.ifc_class
    if element.ifc_class:
        ctx.warn(
            f"{element.kind.value} {element.id}: ifc_class {element.ifc_class!r} not supported; "
            f"{default} used"
        )
    return default


def new_product(
    ctx: ExportContext,
    element: Element,
    ifc_class: str,
    name: str,
    *,
    description: str,
    contained: bool = True,
) -> Entity:
    """Create an IFC product for a scene element with identity attributes and containment."""
    product = ifcopenshell.api.root.create_entity(ctx.file, ifc_class=ifc_class, name=name)
    set_guid(ctx, product, element.id)
    product.Description = element.description or description
    if is_a(product, "IfcElement"):
        product.Tag = element.id
    ctx.entities[element.id] = product
    if contained:
        if element.storey_id is None:
            ctx.flag(element.id, "no storey assigned (contained in the building)")
        container = container_of(ctx, element.storey_id, element.building_id)
        ifcopenshell.api.spatial.assign_container(
            ctx.file, products=[product], relating_structure=container
        )
    return product


def _assign_layered_type(ctx: ExportContext, product: Entity, type_entity: Entity) -> Entity:
    """Assign a type (creates the layer set usage) and return the usage entity."""
    ifcopenshell.api.type.assign_type(
        ctx.file, related_objects=[product], relating_type=type_entity
    )
    usage = ifcopenshell.util.element.get_material(product)
    assert usage is not None and is_a(usage, "IfcMaterialLayerSetUsage")
    return usage


def _footprint_local(
    wall: WallGeometry,
) -> list[tuple[float, float]]:
    """Footprint in the wall frame: x along the axis from the trimmed start, y = left normal."""
    ax, ay = wall.axis_start
    dx, dy = wall.direction
    points: list[tuple[float, float]] = []
    for px, py in wall.footprint:
        rx, ry = px - ax, py - ay
        points.append((rx * dx + ry * dy, -rx * dy + ry * dx))
    return points


def _is_rectangle(points: list[tuple[float, float]], length: float, lo: float, hi: float) -> bool:
    expected = [(0.0, lo), (length, lo), (length, hi), (0.0, hi)]
    return all(
        math.hypot(p[0] - e[0], p[1] - e[1]) <= RECTANGLE_TOLERANCE
        for p, e in zip(points, expected, strict=True)
    )


def write_walls(ctx: ExportContext) -> None:
    """IfcWall per scene wall with type, layer set usage, axis and body."""
    f = ctx.file
    for element in ctx.scene.elements.values():
        if not isinstance(element, WallElement):
            continue
        geometry = ctx.geometry.walls.get(element.id)
        if geometry is None:
            ctx.flag(element.id, "no geometry: wall not exported")
            continue
        if geometry.length <= 1e-6 or geometry.gross_footprint_area <= 0.0:
            # a stub that the join cuts consume completely would become a zero-size profile (invalid IFC)
            ctx.flag(element.id, "wall has no positive extent after join cuts: not exported")
            continue
        ifc_class = ifc_class_for(ctx, element, "IfcWall")
        predefined = valid_predefined(ctx, element, "IfcWallTypeEnum", "STANDARD")
        wall = new_product(
            ctx,
            element,
            ifc_class,
            element_name(element, "Wall"),
            description=identity.wall_description(
                element.params.function.value, element.params.thickness, geometry.height
            ),
        )

        scene_type = ctx.scene.types.get(element.type_id or "")
        wall_type = layered_type(
            ctx,
            ifc_class="IfcWallType",
            predefined_type=predefined,
            base_name=scene_type.name if scene_type and scene_type.name else "Wall",
            scene_type_id=element.type_id,
            layers=element.params.layers,
        )
        usage = _assign_layered_type(ctx, wall, wall_type)
        # Pitfall 3: the usage offset must equal the body offset (lower face along local +Y).
        ifcopenshell.api.material.edit_layer_usage(
            f, usage=usage, attributes={"OffsetFromReferenceLine": float(geometry.offset_right)}
        )
        wall.PredefinedType = predefined
        wall.ObjectType = str(wall_type.Name)

        dx, dy = geometry.direction
        shapes.place(
            ctx,
            wall,
            (geometry.axis_start[0], geometry.axis_start[1], geometry.base_z),
            x_axis=(dx, dy, 0.0),
        )
        local = _footprint_local(geometry)
        lo, hi = geometry.offset_right, geometry.offset_left
        if _is_rectangle(local, geometry.length, lo, hi):
            profile = shapes.rectangle_profile(
                f, (geometry.length / 2.0, (lo + hi) / 2.0), geometry.length, hi - lo
            )
        else:
            profile = shapes.polygon_profile(f, local)
        solid = shapes.extrusion(f, profile, geometry.height)
        shapes.assign_representations(
            ctx,
            wall,
            [
                shapes.axis_representation(ctx, geometry.length),
                shapes.swept_solid_representation(ctx, [solid]),
            ],
        )


def write_joins(ctx: ExportContext) -> None:
    """IfcRelConnectsPathElements for every wall join without geometry issues."""
    for element in ctx.scene.elements.values():
        if not isinstance(element, WallJoinElement):
            continue
        issues = ctx.geometry.issues.get(element.id, ())
        wall_a = ctx.entities.get(element.params.wall_a)
        wall_b = ctx.entities.get(element.params.wall_b)
        if issues or wall_a is None or wall_b is None:
            ctx.flag(element.id, "wall join not exported: " + ("; ".join(issues) or "unknown wall"))
            continue
        ifcopenshell.api.geometry.connect_path(
            ctx.file,
            relating_element=wall_a,
            related_element=wall_b,
            relating_connection=JOIN_END[element.params.end_a],
            related_connection=JOIN_END[element.params.end_b],
            description=element_name(element, "Join"),
        )


def _local_ring(
    points: tuple[tuple[float, float], ...], origin: tuple[float, float]
) -> list[tuple[float, float]]:
    return [(x - origin[0], y - origin[1]) for x, y in points]


def write_slabs(ctx: ExportContext) -> None:
    """IfcSlab (FLOOR or BASESLAB) with type, layer set usage and a profile with holes."""
    f = ctx.file
    for element in ctx.scene.elements.values():
        if not isinstance(element, SlabElement):
            continue
        geometry: SlabGeometry | None = ctx.geometry.slabs.get(element.id)
        if geometry is None:
            ctx.flag(element.id, "no geometry: slab not exported")
            continue
        predefined = valid_predefined(ctx, element, "IfcSlabTypeEnum", "FLOOR")
        slab = new_product(
            ctx,
            element,
            "IfcSlab",
            element_name(element, "Slab"),
            description=identity.slab_description(predefined, geometry.thickness),
        )
        scene_type = ctx.scene.types.get(element.type_id or "")
        slab_type = layered_type(
            ctx,
            ifc_class="IfcSlabType",
            predefined_type=predefined,
            base_name=scene_type.name if scene_type and scene_type.name else "Slab",
            scene_type_id=element.type_id,
            layers=element.params.layers,
        )
        usage = _assign_layered_type(ctx, slab, slab_type)
        # The body extends downward from the placement (top face): layers run in -Z.
        ifcopenshell.api.material.edit_layer_usage(
            f,
            usage=usage,
            attributes={"DirectionSense": "NEGATIVE", "OffsetFromReferenceLine": 0.0},
        )
        slab.PredefinedType = predefined
        slab.ObjectType = str(slab_type.Name)

        origin = (min(p[0] for p in geometry.outline), min(p[1] for p in geometry.outline))
        shapes.place(ctx, slab, (origin[0], origin[1], geometry.top_z))
        profile = shapes.polygon_profile(
            f,
            _local_ring(geometry.outline, origin),
            [_local_ring(hole, origin) for hole in geometry.holes],
        )
        solid = shapes.extrusion(f, profile, geometry.thickness, direction=(0.0, 0.0, -1.0))
        shapes.assign_representations(ctx, slab, [shapes.swept_solid_representation(ctx, [solid])])


def _ceiling_frame(
    geometry: CeilingGeometry,
) -> tuple[
    tuple[float, float, float],
    tuple[float, float, float],
    tuple[float, float, float],
    list[tuple[float, float]],
    tuple[float, float, float],
]:
    """Placement origin, X axis, Z axis, local outline and local extrusion direction."""
    outline = geometry.outline
    if geometry.plane_normal is None:
        origin2 = (min(p[0] for p in outline), min(p[1] for p in outline))
        origin = (origin2[0], origin2[1], geometry.elevation)
        return (
            origin,
            (1.0, 0.0, 0.0),
            (0.0, 0.0, 1.0),
            _local_ring(outline, origin2),
            (0.0, 0.0, 1.0),
        )

    nx, ny, nz = geometry.plane_normal
    length = math.sqrt(nx * nx + ny * ny + nz * nz)
    z_axis = (nx / length, ny / length, nz / length)
    horizontal = math.hypot(z_axis[0], z_axis[1])
    # X' = z_axis x ez is horizontal and lies in the plane; Y' = Z' x X'.
    x_axis = (z_axis[1] / horizontal, -z_axis[0] / horizontal, 0.0)
    y_axis = (
        z_axis[1] * x_axis[2] - z_axis[2] * x_axis[1],
        z_axis[2] * x_axis[0] - z_axis[0] * x_axis[2],
        z_axis[0] * x_axis[1] - z_axis[1] * x_axis[0],
    )
    origin = (outline[0][0], outline[0][1], geometry.outline_z[0])
    local: list[tuple[float, float]] = []
    for (px, py), pz in zip(outline, geometry.outline_z, strict=True):
        d = (px - origin[0], py - origin[1], pz - origin[2])
        local.append(
            (
                d[0] * x_axis[0] + d[1] * x_axis[1] + d[2] * x_axis[2],
                d[0] * y_axis[0] + d[1] * y_axis[1] + d[2] * y_axis[2],
            )
        )
    direction = (x_axis[2], y_axis[2], z_axis[2])
    return origin, x_axis, z_axis, local, direction


def write_ceilings(ctx: ExportContext) -> None:
    """Ceilings as IfcSlab (USERDEFINED / CEILING) or IfcCovering (CEILING), per the settings."""
    f = ctx.file
    as_covering = ctx.scene.settings.ceiling_ifc_class == CeilingIfcClass.COVERING
    for element in ctx.scene.elements.values():
        if not isinstance(element, CeilingElement):
            continue
        geometry: CeilingGeometry | None = ctx.geometry.ceilings.get(element.id)
        if geometry is None:
            ctx.flag(element.id, "no geometry: ceiling not exported")
            continue
        scene_type = ctx.scene.types.get(element.type_id or "")
        base = scene_type.name if scene_type and scene_type.name else "Ceiling"
        layers = [MaterialLayer(material="Unspecified", thickness=geometry.thickness)]
        ceiling_text = identity.ceiling_description(geometry.thickness)
        if as_covering:
            ceiling = new_product(
                ctx,
                element,
                "IfcCovering",
                element_name(element, "Ceiling"),
                description=ceiling_text,
            )
            ceiling_type = layered_type(
                ctx,
                ifc_class="IfcCoveringType",
                predefined_type="CEILING",
                base_name=base,
                scene_type_id=element.type_id,
                layers=layers,
            )
        else:
            ceiling = new_product(
                ctx,
                element,
                "IfcSlab",
                element_name(element, "Ceiling"),
                description=ceiling_text,
            )
            ceiling_type = layered_type(
                ctx,
                ifc_class="IfcSlabType",
                predefined_type="USERDEFINED",
                base_name=base,
                scene_type_id=element.type_id,
                layers=layers,
                element_type="CEILING",
            )
        _assign_layered_type(ctx, ceiling, ceiling_type)
        # assign_type clears PredefinedType and ObjectType on the occurrence: set them again.
        if as_covering:
            ceiling.PredefinedType = "CEILING"
            ceiling.ObjectType = str(ceiling_type.Name)
        else:
            ceiling.PredefinedType = "USERDEFINED"
            ceiling.ObjectType = "CEILING"

        origin, x_axis, z_axis, local, direction = _ceiling_frame(geometry)
        shapes.place(ctx, ceiling, origin, x_axis=x_axis, z_axis=z_axis)
        profile = shapes.polygon_profile(f, local)
        solid = shapes.extrusion(f, profile, geometry.thickness, direction=direction)
        shapes.assign_representations(
            ctx, ceiling, [shapes.swept_solid_representation(ctx, [solid])]
        )


def _opening_frame(
    geometry: OpeningGeometry,
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """Placement origin and X axis of the opening frame (Y = left normal, Z up)."""
    return geometry.origin, geometry.u_axis


def write_openings(ctx: ExportContext) -> None:
    """IfcOpeningElement per opening, voiding its host wall (IfcRelVoidsElement)."""
    f = ctx.file
    for element in ctx.scene.elements.values():
        if not isinstance(element, OpeningElement):
            continue
        geometry = ctx.geometry.openings.get(element.id)
        wall = ctx.entities.get(element.params.host_wall_id)
        if geometry is None or wall is None:
            ctx.flag(element.id, "host wall or geometry missing: opening not exported")
            continue
        predefined = valid_predefined(ctx, element, "IfcOpeningElementTypeEnum", "OPENING")
        opening = new_product(
            ctx,
            element,
            "IfcOpeningElement",
            element_name(element, "Opening"),
            description=identity.opening_description(
                geometry.width, geometry.height, str(wall.Name)
            ),
            contained=False,
        )
        opening.PredefinedType = predefined
        opening.ObjectType = identity.OPENING_OBJECT_TYPE
        ifcopenshell.api.feature.add_feature(f, feature=opening, element=wall)
        origin, x_axis = _opening_frame(geometry)
        shapes.place(ctx, opening, origin, x_axis=x_axis)
        profile = shapes.rectangle_profile(
            f, (geometry.width / 2.0, -geometry.depth / 2.0), geometry.width, geometry.depth
        )
        solid = shapes.extrusion(f, profile, geometry.height)
        shapes.assign_representations(
            ctx, opening, [shapes.swept_solid_representation(ctx, [solid])]
        )


def _filling_solids(
    ctx: ExportContext,
    *,
    is_door: bool,
    width: float,
    height: float,
    thickness: float,
    frame_depth: float,
    mullions: Sequence[float] = (),
    glazed: bool = False,
) -> tuple[list[Entity], list[Entity]]:
    """Body items of a window or door in the opening frame: ``(opaque items, glass items)``.

    * Window: lining (hollow rectangle), one bar per mullion and one glass pane per pane between the
      lining and the bars - every part is its own item so a viewer shows the panes.
    * Door: lining, leaf; a glazed door has a kick panel and a glass panel above it instead of the full leaf.
    """
    f = ctx.file
    lining = min(LINING_THICKNESS, width / 4.0, height / 4.0)
    depth = min(frame_depth, thickness)
    axis = (0.0, -1.0, 0.0)
    ref = (1.0, 0.0, 0.0)
    center = (width / 2.0, height / 2.0)

    def solid(profile: Entity, solid_depth: float) -> Entity:
        y0 = -(thickness - solid_depth) / 2.0
        return shapes.extrusion(
            f, profile, solid_depth, origin=(0.0, y0, 0.0), axis=axis, ref_direction=ref
        )

    glass_depth = min(GLAZING_THICKNESS, depth)
    if is_door:
        frame_profile = shapes.polygon_profile(
            f,
            [
                (0.0, 0.0),
                (lining, 0.0),
                (lining, height - lining),
                (width - lining, height - lining),
                (width - lining, 0.0),
                (width, 0.0),
                (width, height),
                (0.0, height),
            ],
        )
        panel_depth = min(PANEL_DEPTH, depth)
        if not glazed:
            panel_profile = shapes.rectangle_profile(
                f, (width / 2.0, (height - lining) / 2.0), width - 2.0 * lining, height - lining
            )
            return [solid(frame_profile, depth), solid(panel_profile, panel_depth)], []
        kick = min(0.45, 0.25 * height)
        kick_profile = shapes.rectangle_profile(f, (width / 2.0, kick / 2.0), width - 2.0 * lining, kick)
        glass_top = height - 2.0 * lining
        glass_height = glass_top - kick
        glass_profile = shapes.rectangle_profile(
            f, (width / 2.0, kick + glass_height / 2.0), width - 4.0 * lining, glass_height
        )
        return [solid(frame_profile, depth), solid(kick_profile, panel_depth)], [solid(glass_profile, glass_depth)]
    frame_profile = shapes.hollow_rectangle_profile(f, center, width, height, lining)
    opaque = [solid(frame_profile, depth)]
    bars = sorted(min(max(float(value), lining + MULLION_WIDTH / 2.0), width - lining - MULLION_WIDTH / 2.0) for value in mullions)
    inner_height = height - 2.0 * lining
    for position in bars:
        bar_profile = shapes.rectangle_profile(f, (position, height / 2.0), MULLION_WIDTH, inner_height)
        opaque.append(solid(bar_profile, min(PANEL_DEPTH, depth)))
    edges = [lining] + [edge for position in bars for edge in (position - MULLION_WIDTH / 2.0, position + MULLION_WIDTH / 2.0)] + [width - lining]
    glass: list[Entity] = []
    for index in range(0, len(edges), 2):
        left, right = edges[index], edges[index + 1]
        if right - left < 0.02:
            continue
        pane_profile = shapes.rectangle_profile(f, ((left + right) / 2.0, height / 2.0), right - left, inner_height)
        glass.append(solid(pane_profile, glass_depth))
    return opaque, glass


def _style(ctx: ExportContext, key: str, colour: tuple[float, float, float], transparency: float) -> Entity:
    """Shading surface style (cached per file) for the window frame / glass items."""
    cache_key = ("filling-style", key)
    if cache_key in ctx.types:
        return ctx.types[cache_key]
    style = ifcopenshell.api.style.add_style(ctx.file, name=f"SAMcloud {key}")
    ifcopenshell.api.style.add_surface_style(
        ctx.file,
        style=style,
        ifc_class="IfcSurfaceStyleShading",
        attributes={
            "SurfaceColour": {"Name": None, "Red": colour[0], "Green": colour[1], "Blue": colour[2]},
            "Transparency": transparency,
        },
    )
    ctx.types[cache_key] = style
    return style


def write_fillings(ctx: ExportContext) -> None:
    """IfcWindow and IfcDoor with types, filling their opening (IfcRelFillsElement)."""
    f = ctx.file
    for element in ctx.scene.elements.values():
        if not isinstance(element, WindowElement | DoorElement):
            continue
        is_door = isinstance(element, DoorElement)
        pool = ctx.geometry.doors if is_door else ctx.geometry.windows
        filling: FillingGeometry | None = pool.get(element.id)
        opening = ctx.entities.get(element.params.opening_id)
        opening_geometry = ctx.geometry.openings.get(element.params.opening_id)
        label = "Door" if is_door else "Window"
        if filling is None or opening is None or opening_geometry is None:
            ctx.flag(element.id, f"opening or geometry missing: {label.lower()} not exported")
            continue
        ifc_class = "IfcDoor" if is_door else "IfcWindow"
        enum_name = "IfcDoorTypeEnum" if is_door else "IfcWindowTypeEnum"
        predefined = valid_predefined(ctx, element, enum_name, label.upper())
        width, height = filling.width, filling.height
        if isinstance(element, DoorElement):
            text = identity.door_description(width, height, element.params.operation.value)
        else:
            text = identity.window_description(width, height, filling.sill_height)
        product = new_product(
            ctx, element, ifc_class, element_name(element, label), description=text
        )
        scene_type = ctx.scene.types.get(element.type_id or "")
        base = scene_type.name if scene_type and scene_type.name else label
        if isinstance(element, DoorElement):
            type_entity = door_type(ctx, element, width, height, base)
        else:
            assert isinstance(element, WindowElement)
            type_entity = window_type(ctx, element, width, height, base)
        ifcopenshell.api.type.assign_type(
            f,
            related_objects=[product],
            relating_type=type_entity,
            should_map_representations=False,
        )
        product.PredefinedType = predefined
        product.ObjectType = str(type_entity.Name)
        product.OverallWidth = float(width)
        product.OverallHeight = float(height)
        if isinstance(element, DoorElement):
            operation, user_defined = door_operation(element)
            product.OperationType = operation
            product.UserDefinedOperationType = user_defined
        else:
            assert isinstance(element, WindowElement)
            partitioning, user_defined = window_partitioning(window_pane_count(element))
            product.PartitioningType = partitioning
            if user_defined is not None:
                product.UserDefinedPartitioningType = user_defined
        ifcopenshell.api.feature.add_filling(f, opening=opening, element=product)
        origin, x_axis = _opening_frame(opening_geometry)
        shapes.place(ctx, product, origin, x_axis=x_axis)
        frame_depth = element.params.frame_depth or DEFAULT_FRAME_DEPTH
        opaque, glass = _filling_solids(
            ctx,
            is_door=is_door,
            width=width,
            height=height,
            thickness=opening_geometry.depth,
            frame_depth=frame_depth,
            mullions=() if is_door else element.params.mullion_positions,
            glazed=is_door and element.params.glazed,
        )
        shapes.assign_representations(
            ctx, product, [shapes.swept_solid_representation(ctx, [*opaque, *glass])]
        )
        frame_style = _style(ctx, "window frame" if not is_door else "door leaf", (0.93, 0.93, 0.91) if not is_door else (0.55, 0.30, 0.15), 0.0)
        for item in opaque:
            ifcopenshell.api.style.assign_item_style(f, item=item, style=frame_style)
        if glass:
            glass_style = _style(ctx, "glass", (0.65, 0.82, 0.92), 0.65)
            for item in glass:
                ifcopenshell.api.style.assign_item_style(f, item=item, style=glass_style)
