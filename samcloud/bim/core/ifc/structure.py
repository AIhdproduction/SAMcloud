"""Measured structure: IfcColumn and IfcBeam with their types, profiles, standard psets and base quantities.

* Column: ``IfcRectangleProfileDef`` (rotated by ``rotation``) or ``IfcCircleProfileDef`` extruded upward from ``base_z``
  by ``height``; placement at the column centre on its base.
* Beam: ``IfcRectangleProfileDef`` (width x depth) extruded along the beam axis; placement at the axis start on the
  underside, local X along the axis.
* One ``IfcColumnType`` / ``IfcBeamType`` per (predefined type, profile, rounded size); ``Pset_ColumnCommon`` /
  ``Pset_BeamCommon`` (Reference) and ``Qto_ColumnBaseQuantities`` / ``Qto_BeamBaseQuantities``.

Raumlot_Quality / Raumlot_Location and the user properties are written for every element by
:func:`samcloud.bim.core.ifc.properties.write_properties`.
"""

from __future__ import annotations

import math

import ifcopenshell
import ifcopenshell.api.pset
import ifcopenshell.api.root
import ifcopenshell.api.style
import ifcopenshell.api.type

from samcloud.bim.core.ifc import shapes
from samcloud.bim.core.ifc.context import ExportContext
from samcloud.bim.core.ifc.elements import element_name, new_product, valid_predefined
from samcloud.bim.core.ifc.ids import set_key_guid
from samcloud.bim.core.model import BeamElement, ColumnElement

Entity = ifcopenshell.entity_instance
_COLOUR = (0.62, 0.62, 0.60)


def _style(ctx: ExportContext) -> Entity:
    key = ("structure-style",)
    if key in ctx.types:
        return ctx.types[key]
    style = ifcopenshell.api.style.add_style(ctx.file, name="SAMcloud structure")
    ifcopenshell.api.style.add_surface_style(
        ctx.file, style=style, ifc_class="IfcSurfaceStyleShading",
        attributes={"SurfaceColour": {"Name": None, "Red": _COLOUR[0], "Green": _COLOUR[1], "Blue": _COLOUR[2]}, "Transparency": 0.0},
    )
    ctx.types[key] = style
    return style


def _type(ctx: ExportContext, ifc_class: str, predefined: str, name: str) -> Entity:
    key = ("structure-type", ifc_class, predefined, name)
    if key in ctx.types:
        return ctx.types[key]
    entity = ifcopenshell.api.root.create_entity(ctx.file, ifc_class=f"{ifc_class}Type", predefined_type=predefined, name=name)
    set_key_guid(ctx, entity, f"type/{key!r}")
    pset = ifcopenshell.api.pset.add_pset(ctx.file, product=entity, name=f"Pset_{ifc_class[3:]}TypeCommon")
    ifcopenshell.api.pset.edit_pset(ctx.file, pset=pset, properties={"Reference": name})
    ctx.types[key] = entity
    return entity


def _finish(ctx: ExportContext, product: Entity, type_entity: Entity, predefined: str, solid: Entity, pset: str, qto: str, quantities: dict[str, float], reference: str) -> None:
    ifcopenshell.api.type.assign_type(ctx.file, related_objects=[product], relating_type=type_entity, should_map_representations=False)
    product.PredefinedType = predefined
    body = shapes.swept_solid_representation(ctx, [solid])
    ifcopenshell.api.style.assign_representation_styles(ctx.file, shape_representation=body, styles=[_style(ctx)])
    shapes.assign_representations(ctx, product, [body])
    common = ifcopenshell.api.pset.add_pset(ctx.file, product=product, name=pset)
    ifcopenshell.api.pset.edit_pset(ctx.file, pset=common, properties={"Reference": reference})
    quantity_set = ifcopenshell.api.pset.add_qto(ctx.file, product=product, name=qto)
    ifcopenshell.api.pset.edit_qto(ctx.file, qto=quantity_set, properties=quantities)


def write_columns(ctx: ExportContext) -> int:
    f = ctx.file
    written = 0
    for element in ctx.scene.elements.values():
        if not isinstance(element, ColumnElement):
            continue
        params = element.params
        predefined = valid_predefined(ctx, element, "IfcColumnTypeEnum", "COLUMN")
        if params.profile == "circle":
            diameter = params.width
            reference = f"Column D{diameter * 1000:.0f}"
            profile = f.createIfcCircleProfileDef("AREA", None, f.createIfcAxis2Placement2D(f.createIfcCartesianPoint((0.0, 0.0)), None), float(diameter / 2.0))
            area = math.pi * diameter * diameter / 4.0
        else:
            depth = params.depth or params.width
            reference = f"Column {params.width * 1000:.0f}x{depth * 1000:.0f}"
            profile = shapes.rectangle_profile(f, (0.0, 0.0), params.width, depth)
            area = params.width * depth
        product = new_product(ctx, element, "IfcColumn", element_name(element, "Column"), description=f"{reference}, h {params.height:.2f} m")
        angle = math.radians(params.rotation)
        shapes.place(ctx, product, (float(params.center[0]), float(params.center[1]), float(params.base_z)), x_axis=(math.cos(angle), math.sin(angle), 0.0))
        solid = shapes.extrusion(f, profile, params.height)
        _finish(ctx, product, _type(ctx, "IfcColumn", predefined, reference), predefined, solid, "Pset_ColumnCommon", "Qto_ColumnBaseQuantities",
                {"Length": float(params.height), "CrossSectionArea": float(area), "GrossVolume": float(area * params.height)}, reference)
        written += 1
    return written


def write_beams(ctx: ExportContext) -> int:
    f = ctx.file
    written = 0
    for element in ctx.scene.elements.values():
        if not isinstance(element, BeamElement):
            continue
        params = element.params
        predefined = valid_predefined(ctx, element, "IfcBeamTypeEnum", "BEAM")
        dx, dy = params.end[0] - params.start[0], params.end[1] - params.start[1]
        length = math.hypot(dx, dy)
        reference = f"Beam {params.width * 1000:.0f}x{params.depth * 1000:.0f}"
        product = new_product(ctx, element, "IfcBeam", element_name(element, "Beam"), description=f"{reference}, l {length:.2f} m")
        shapes.place(ctx, product, (float(params.start[0]), float(params.start[1]), float(params.bottom_z)), x_axis=(dx / length, dy / length, 0.0))
        # profile in the plane normal to the axis (local Y = across, Z = up): centred across, standing on the underside
        profile = shapes.rectangle_profile(f, (0.0, params.depth / 2.0), params.width, params.depth)
        solid = shapes.extrusion(f, profile, length, axis=(1.0, 0.0, 0.0), ref_direction=(0.0, 1.0, 0.0))
        area = params.width * params.depth
        _finish(ctx, product, _type(ctx, "IfcBeam", predefined, reference), predefined, solid, "Pset_BeamCommon", "Qto_BeamBaseQuantities",
                {"Length": float(length), "CrossSectionArea": float(area), "GrossVolume": float(area * length)}, reference)
        written += 1
    return written


def write_structure(ctx: ExportContext) -> None:
    write_columns(ctx)
    write_beams(ctx)
