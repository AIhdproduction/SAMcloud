"""Project, units, representation contexts and the spatial tree (site, building, storeys)."""

from __future__ import annotations

import ifcopenshell
import ifcopenshell.api.aggregate
import ifcopenshell.api.context
import ifcopenshell.api.root
import ifcopenshell.api.unit

from samcloud.bim.core.ifc.context import ExportContext
from samcloud.bim.core.ifc.ids import set_guid, set_key_guid
from samcloud.bim.core.ifc.shapes import place
from samcloud.bim.core.model import BuildingElement

Entity = ifcopenshell.entity_instance

DEFAULT_SITE = "Site"
DEFAULT_BUILDING = "Building"
DEFAULT_STOREY = "Storey"


def create_project(ctx: ExportContext) -> None:
    """IfcProject with explicit SI units (metre, square metre, cubic metre, radian) and contexts.

    ``unit.assign_unit(f)`` without arguments would assign MILLIMETRE, so the units are passed.
    """
    f = ctx.file
    ctx.project = ifcopenshell.api.root.create_entity(f, ifc_class="IfcProject", name="SAMcloud")
    set_key_guid(ctx, ctx.project, "project")
    units = [
        ifcopenshell.api.unit.add_si_unit(f, unit_type="LENGTHUNIT"),
        ifcopenshell.api.unit.add_si_unit(f, unit_type="AREAUNIT"),
        ifcopenshell.api.unit.add_si_unit(f, unit_type="VOLUMEUNIT"),
        ifcopenshell.api.unit.add_si_unit(f, unit_type="PLANEANGLEUNIT"),
    ]
    ifcopenshell.api.unit.assign_unit(f, units=units)
    # assign_unit stores the units in a set: restore a fixed order for deterministic files.
    ctx.project.UnitsInContext.Units = units

    model = ifcopenshell.api.context.add_context(f, context_type="Model")
    ctx.model_context = model
    ctx.body_context = ifcopenshell.api.context.add_context(
        f,
        context_type="Model",
        context_identifier="Body",
        target_view="MODEL_VIEW",
        parent=model,
    )
    ctx.axis_context = ifcopenshell.api.context.add_context(
        f,
        context_type="Model",
        context_identifier="Axis",
        target_view="GRAPH_VIEW",
        parent=model,
    )
    plan = ifcopenshell.api.context.add_context(f, context_type="Plan")
    ifcopenshell.api.context.add_context(
        f,
        context_type="Plan",
        context_identifier="FootPrint",
        target_view="PLAN_VIEW",
        parent=plan,
    )


def create_spatial_tree(ctx: ExportContext) -> None:
    """IfcSite, IfcBuilding and IfcBuildingStorey with names, elevations and placements.

    A scene without site, building or storey gets a default site or building so that every
    element has a parent. Storeys without ``building_id`` belong to the first building.
    """
    f = ctx.file
    scene = ctx.scene
    assert ctx.project is not None

    site_items = [(site.id, site.name or DEFAULT_SITE) for site in scene.sites.values()]
    if not site_items:
        site_items = [("default-site", DEFAULT_SITE)]
    for site_id, name in site_items:
        site = ifcopenshell.api.root.create_entity(f, ifc_class="IfcSite", name=name)
        site.CompositionType = "ELEMENT"
        set_guid(ctx, site, site_id)
        ifcopenshell.api.aggregate.assign_object(f, products=[site], relating_object=ctx.project)
        place(ctx, site, (0.0, 0.0, 0.0))
        ctx.sites[site_id] = site
    ctx.default_site_id = site_items[0][0]

    building_items = [
        (b.id, b.name or DEFAULT_BUILDING, b.site_id) for b in scene.buildings.values()
    ]
    building_items.extend(
        (element.id, element.name or DEFAULT_BUILDING, element.site_id)
        for element in scene.elements.values()
        if isinstance(element, BuildingElement) and element.id not in scene.buildings
    )
    if not building_items:
        building_items = [("default-building", DEFAULT_BUILDING, None)]
    for building_id, name, site_ref in building_items:
        building = ifcopenshell.api.root.create_entity(f, ifc_class="IfcBuilding", name=name)
        building.CompositionType = "ELEMENT"
        set_guid(ctx, building, building_id)
        parent = ctx.sites.get(site_ref or "", ctx.sites[ctx.default_site_id])
        ifcopenshell.api.aggregate.assign_object(f, products=[building], relating_object=parent)
        place(ctx, building, (0.0, 0.0, 0.0))
        ctx.buildings[building_id] = building
    ctx.default_building_id = building_items[0][0]

    for storey_item in scene.storeys.values():
        storey = ifcopenshell.api.root.create_entity(
            f, ifc_class="IfcBuildingStorey", name=storey_item.name or DEFAULT_STOREY
        )
        storey.CompositionType = "ELEMENT"
        storey.Elevation = float(storey_item.elevation)
        set_guid(ctx, storey, storey_item.id)
        parent = ctx.buildings.get(
            storey_item.building_id or "", ctx.buildings[ctx.default_building_id]
        )
        ifcopenshell.api.aggregate.assign_object(f, products=[storey], relating_object=parent)
        place(ctx, storey, (0.0, 0.0, float(storey_item.elevation)))
        ctx.storeys[storey_item.id] = storey


def container_of(ctx: ExportContext, storey_id: str | None, building_id: str | None) -> Entity:
    """Spatial parent of an element: its storey, else its building (outdoor elements)."""
    if storey_id is not None and storey_id in ctx.storeys:
        return ctx.storeys[storey_id]
    if building_id is not None and building_id in ctx.buildings:
        return ctx.buildings[building_id]
    return ctx.buildings[ctx.default_building_id]
