"""Stairs: IfcStair with IfcStairType, aggregated IfcStairFlight and landing IfcSlab (LANDING).

Decisions (see docs/IFC_MAPPING.md, "Stairs (WP5.3)"):

* The IfcStair has no own body; it aggregates one IfcStairFlight per flight and one IfcSlab with
  PredefinedType LANDING per landing (IfcRelAggregates). Only the IfcStair is contained in the
  storey.
* Stair placement: origin at the foot of the first riser at the base elevation, local X along the
  travel direction, local Y the left normal. Flights and landings are placed globally (their
  relative placements are derived).
* Flight body: the stepped side profile (s, z) extruded across the width (extrusion along -Y from
  the left edge). Landing body: plan outline extruded downward from the landing top.
* PredefinedType: one flight STRAIGHT_RUN_STAIR, two flights TWO_STRAIGHT_RUN_STAIR, more
  USERDEFINED (ObjectType carries the type name).
* A ``non_parametric`` stair (fallback of the extraction) is written the same way as a simplified
  solid and carries ``Raumlot_Quality.NonParametric = True`` plus its notes.
"""

from __future__ import annotations

import ifcopenshell
import ifcopenshell.api.aggregate
import ifcopenshell.api.pset
import ifcopenshell.api.root
import ifcopenshell.api.type

from samcloud.bim.core.geometry import FlightGeometry, LandingGeometry, StairGeometry
from samcloud.bim.core.ifc import identity, shapes
from samcloud.bim.core.ifc.context import ExportContext
from samcloud.bim.core.ifc.elements import element_name, enum_items, new_product
from samcloud.bim.core.ifc.ids import set_key_guid
from samcloud.bim.core.ifc.types import mm_text
from samcloud.bim.core.model import StairElement

Entity = ifcopenshell.entity_instance


def stair_predefined(ctx: ExportContext, element: StairElement, flights: int) -> str:
    """The element's valid IfcStairTypeEnum literal, else derived from the flight count."""
    derived = {1: "STRAIGHT_RUN_STAIR", 2: "TWO_STRAIGHT_RUN_STAIR"}.get(flights, "USERDEFINED")
    value = element.predefined_type
    if value is None:
        return derived
    if value in enum_items(ctx, "IfcStairTypeEnum"):
        return value
    ctx.flag(element.id, f"invalid PredefinedType {value!r} for IfcStairTypeEnum; {derived} used")
    return derived


def stair_type(ctx: ExportContext, element: StairElement, predefined: str) -> Entity:
    """IfcStairType, one per distinct (predefined type, risers, riser height, tread, width)."""
    params = element.params
    key: tuple[object, ...] = (
        "IfcStairType",
        predefined,
        element.type_id,
        params.step_count,
        round(params.riser_height, 6),
        round(params.tread_length, 6),
        round(params.width, 6),
    )
    if key in ctx.types:
        return ctx.types[key]
    scene_type = ctx.scene.types.get(element.type_id or "")
    base = scene_type.name if scene_type and scene_type.name else "Stair"
    name = (
        f"{base} {params.step_count}x{mm_text(params.riser_height)}x"
        f"{mm_text(params.tread_length)} w{mm_text(params.width)}"
    )
    entity = ifcopenshell.api.root.create_entity(
        ctx.file, ifc_class="IfcStairType", predefined_type=predefined, name=name
    )
    set_key_guid(ctx, entity, f"type/{key!r}")
    ctx.types[key] = entity
    return entity


def _count(ctx: ExportContext, value: int) -> Entity:
    return ctx.file.create_entity("IfcCountMeasure", int(value))


def _positive(ctx: ExportContext, value: float) -> Entity:
    return ctx.file.create_entity("IfcPositiveLengthMeasure", float(value))


def _add_pset(ctx: ExportContext, product: Entity, name: str, props: dict[str, object]) -> None:
    pset = ifcopenshell.api.pset.add_pset(ctx.file, product=product, name=name)
    ifcopenshell.api.pset.edit_pset(ctx.file, pset=pset, properties=props)


def _write_flight(
    ctx: ExportContext,
    element: StairElement,
    stair: Entity,
    geometry: StairGeometry,
    flight: FlightGeometry,
) -> Entity:
    f = ctx.file
    entity = ifcopenshell.api.root.create_entity(
        f,
        ifc_class="IfcStairFlight",
        predefined_type="STRAIGHT",
        name=f"{stair.Name} flight {flight.index + 1}",
    )
    set_key_guid(ctx, entity, f"{element.id}/flight/{flight.index}")
    entity.Description = (
        f"Stair flight {flight.index + 1}, {flight.riser_count} risers, "
        f"{flight.tread_count} treads, rise {identity.metres(flight.rise)} m"
    )
    entity.ObjectType = "Flight"
    entity.Tag = f"{element.id}/flight/{flight.index}"
    entity.NumberOfRisers = flight.riser_count
    entity.NumberOfTreads = flight.tread_count
    entity.RiserHeight = float(flight.riser_height)
    entity.TreadLength = float(flight.tread_length)
    ifcopenshell.api.aggregate.assign_object(f, products=[entity], relating_object=stair)
    dx, dy = geometry.direction
    shapes.place(ctx, entity, flight.origin, x_axis=(dx, dy, 0.0))
    profile = shapes.polygon_profile(f, list(flight.profile))
    solid = shapes.extrusion(
        f,
        profile,
        geometry.width,
        origin=(0.0, geometry.width / 2.0, 0.0),
        axis=(0.0, -1.0, 0.0),
        ref_direction=(1.0, 0.0, 0.0),
    )
    shapes.assign_representations(ctx, entity, [shapes.swept_solid_representation(ctx, [solid])])
    _add_pset(
        ctx,
        entity,
        "Pset_StairFlightCommon",
        {
            "Reference": f"{element.tag or element.name or element.id[:8]}/{flight.index + 1}",
            "NumberOfRiser": _count(ctx, flight.riser_count),
            "NumberOfTreads": _count(ctx, flight.tread_count),
            "RiserHeight": _positive(ctx, flight.riser_height),
            "TreadLength": _positive(ctx, flight.tread_length),
        },
    )
    qto = ifcopenshell.api.pset.add_qto(f, product=entity, name="Qto_StairFlightBaseQuantities")
    ifcopenshell.api.pset.edit_qto(
        f,
        qto=qto,
        properties={
            "Length": float(flight.length),
            "GrossVolume": float(flight.gross_volume),
            "NetVolume": float(flight.gross_volume),
        },
    )
    return entity


def _write_landing(
    ctx: ExportContext, element: StairElement, stair: Entity, landing: LandingGeometry
) -> Entity:
    f = ctx.file
    entity = ifcopenshell.api.root.create_entity(
        f,
        ifc_class="IfcSlab",
        predefined_type="LANDING",
        name=f"{stair.Name} landing {landing.index + 1}",
    )
    set_key_guid(ctx, entity, f"{element.id}/landing/{landing.index}")
    entity.Description = f"Stair landing {landing.index + 1}, {identity.metres(landing.area)} m2"
    entity.ObjectType = "Landing"
    entity.Tag = f"{element.id}/landing/{landing.index}"
    ifcopenshell.api.aggregate.assign_object(f, products=[entity], relating_object=stair)
    origin = (min(p[0] for p in landing.outline), min(p[1] for p in landing.outline))
    shapes.place(ctx, entity, (origin[0], origin[1], landing.top_z))
    ring = [(x - origin[0], y - origin[1]) for x, y in landing.outline]
    solid = shapes.extrusion(
        f,
        shapes.polygon_profile(f, ring),
        landing.top_z - landing.bottom_z,
        direction=(0.0, 0.0, -1.0),
    )
    shapes.assign_representations(ctx, entity, [shapes.swept_solid_representation(ctx, [solid])])
    return entity


def write_stairs(ctx: ExportContext) -> None:
    """IfcStair with type, aggregated flights and landings for every stair of the scene."""
    for element in ctx.scene.elements.values():
        if not isinstance(element, StairElement):
            continue
        geometry = ctx.geometry.stairs.get(element.id)
        if geometry is None:
            ctx.flag(element.id, "no geometry: stair not exported")
            continue
        predefined = stair_predefined(ctx, element, len(geometry.flights))
        stair = new_product(
            ctx,
            element,
            "IfcStair",
            element_name(element, "Stair"),
            description=identity.stair_description(
                predefined,
                geometry.step_count,
                geometry.riser_height,
                geometry.tread_length,
                geometry.width,
            ),
        )
        type_entity = stair_type(ctx, element, predefined)
        ifcopenshell.api.type.assign_type(
            ctx.file,
            related_objects=[stair],
            relating_type=type_entity,
            should_map_representations=False,
        )
        stair.PredefinedType = predefined
        stair.ObjectType = str(type_entity.Name)
        dx, dy = geometry.direction
        shapes.place(
            ctx,
            stair,
            (geometry.start[0], geometry.start[1], geometry.base_z),
            x_axis=(dx, dy, 0.0),
        )
        for flight in geometry.flights:
            _write_flight(ctx, element, stair, geometry, flight)
        for landing in geometry.landings:
            _write_landing(ctx, element, stair, landing)
