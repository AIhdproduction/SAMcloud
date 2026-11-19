"""Property sets and quantities: standard Pset/Qto, Raumlot_Quality, Raumlot_Location, Raumlot_User.

Quantities come from :mod:`samcloud.bim.core.geometry` only. Standard property names are checked against the
IFC4 templates in the tests.

Raumlot_Quality (every exported element) carries, as separate properties: ElementId, Provenance,
RunId, AiBackend, AiModelId, ToleranceProfile, DataSource, Locked, Status, DeviationRms/P95/Max,
Coverage, Confidence, PointCount (only the metrics that exist), ParameterOrigins (one text
"name=origin; name=origin") plus one property ``Origin_<parameter>`` per parameter origin,
``PropertyOrigin_<name>`` per user property, values without a standard quantity (SillHeight,
SillProjection, SillThickness, RevealDepth, SwingSide, Operation, Justification, BaseOffset,
TopElevation, Elevation, InsideSide), Notes, AiComments, Incomplete and MissingAttributes.

Raumlot_Location (every exported element, never omitted, "not applicable" for missing levels):
SiteName, BuildingName, StoreyName, StoreyElevation, SpaceNames, SpaceNumbers, RoomAssignment,
ReferenceX/Y/Z, BBoxMinX/Y/Z, BBoxMaxX/Y/Z (local coordinates plus the origin offset, i.e. map
coordinates when the export is referenced and the coordinates of the source cloud otherwise),
CrsName (EPSG code, name, or "unreferenced") and HeightSystem. The reference point is the
centre of the bounding box in plan at its lowest z.
"""

from __future__ import annotations

import math
from collections.abc import Mapping

import ifcopenshell
import ifcopenshell.api.pset
import ifcopenshell.util.pset

from samcloud.bim.core.geometry import SpaceGeometry
from samcloud.bim.core.geometry.stairs import plan_corners
from samcloud.bim.core.ifc.context import NOT_APPLICABLE, UNREFERENCED, ExportContext, class_of
from samcloud.bim.core.ifc.georef import crs_label
from samcloud.bim.core.model import (
    CeilingElement,
    CeilingIfcClass,
    DoorElement,
    Element,
    ElementKind,
    OpeningElement,
    PropertyValue,
    SlabElement,
    SpaceElement,
    StairElement,
    Tristate,
    WallElement,
    WallFunction,
    WindowElement,
)

Entity = ifcopenshell.entity_instance
PropValue = object

CUSTOM_USER = "Raumlot_User"
_PSET_TEMPLATES = ifcopenshell.util.pset.PsetQto("IFC4")

BBox = tuple[tuple[float, float, float], tuple[float, float, float]]


def _length(ctx: ExportContext, value: float) -> Entity:
    return ctx.file.create_entity("IfcLengthMeasure", float(value))


def _text(ctx: ExportContext, value: str) -> Entity:
    return ctx.file.create_entity("IfcText", value)


def _add_pset(
    ctx: ExportContext, product: Entity, name: str, props: Mapping[str, PropValue]
) -> None:
    if not props:
        return
    pset = ifcopenshell.api.pset.add_pset(ctx.file, product=product, name=name)
    ifcopenshell.api.pset.edit_pset(ctx.file, pset=pset, properties=dict(props))


def _add_qto(ctx: ExportContext, product: Entity, name: str, props: Mapping[str, float]) -> None:
    if not props:
        return
    qto = ifcopenshell.api.pset.add_qto(ctx.file, product=product, name=name)
    ifcopenshell.api.pset.edit_qto(ctx.file, qto=qto, properties=dict(props))


# ----------------------------------------------------------------------------- bounding boxes


def _bbox_of(points: list[tuple[float, float, float]]) -> BBox:
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    zs = [p[2] for p in points]
    return (min(xs), min(ys), min(zs)), (max(xs), max(ys), max(zs))


def element_bbox(ctx: ExportContext, element: Element) -> BBox | None:
    """Bounding box of an element in local (model frame) coordinates, or None without geometry."""
    geometry = ctx.geometry
    eid = element.id
    if eid in geometry.walls:
        wall = geometry.walls[eid]
        return _bbox_of([(x, y, z) for x, y in wall.footprint for z in (wall.base_z, wall.top_z)])
    if eid in geometry.openings:
        return _bbox_of(list(geometry.openings[eid].corners))
    for pool in (geometry.windows, geometry.doors):
        if eid in pool:
            return _bbox_of(list(pool[eid].corners))
    if eid in geometry.slabs:
        slab = geometry.slabs[eid]
        return _bbox_of([(x, y, z) for x, y in slab.outline for z in (slab.bottom_z, slab.top_z)])
    if eid in geometry.ceilings:
        ceiling = geometry.ceilings[eid]
        points = [
            (x, y, z)
            for (x, y), base in zip(ceiling.outline, ceiling.outline_z, strict=True)
            for z in (base, base + ceiling.thickness)
        ]
        return _bbox_of(points)
    if eid in geometry.stairs:
        stair = geometry.stairs[eid]
        corners = plan_corners(stair)
        return _bbox_of([(x, y, z) for x, y in corners for z in (stair.base_z, stair.top_z)])
    if eid in geometry.spaces:
        space = geometry.spaces[eid]
        return _bbox_of(
            [(x, y, z) for x, y in space.exterior for z in (space.floor_z, space.ceiling_z)]
        )
    return None


# ------------------------------------------------------------------------- spaces of elements


class SpaceRelations:
    """Which spaces an element belongs to and how reliable the assignment is."""

    def __init__(self, ctx: ExportContext) -> None:
        self.ctx = ctx
        self.spaces: list[SpaceElement] = [
            e for e in ctx.scene.elements.values() if isinstance(e, SpaceElement)
        ]
        self.by_element: dict[str, list[str]] = {}
        self.ambiguous: set[str] = set()
        for space in self.spaces:
            geometry = ctx.geometry.spaces.get(space.id)
            self._add(space.id, space.id)
            if geometry is None:
                continue
            if _is_ambiguous(geometry):
                self.ambiguous.add(space.id)
            for related in (
                *geometry.bounding_wall_ids,
                *geometry.floor_ids,
                *geometry.ceiling_ids,
            ):
                self._add(related, space.id)
        for element in ctx.scene.elements.values():
            for space_id in element.space_ids:
                self._add(element.id, space_id)

    def _add(self, element_id: str, space_id: str) -> None:
        listed = self.by_element.setdefault(element_id, [])
        if space_id not in listed:
            listed.append(space_id)

    def of(self, element: Element) -> list[SpaceElement]:
        """Spaces of an element in scene order. Openings, windows, doors use their host wall."""
        ids = list(self.by_element.get(element.id, []))
        host_id: str | None = None
        if isinstance(element, OpeningElement):
            host_id = element.params.host_wall_id
        elif isinstance(element, WindowElement | DoorElement):
            opening = self.ctx.scene.elements.get(element.params.opening_id)
            if isinstance(opening, OpeningElement):
                host_id = opening.params.host_wall_id
        if host_id is not None:
            for space_id in self.by_element.get(host_id, []):
                if space_id not in ids:
                    ids.append(space_id)
        order = {space.id: index for index, space in enumerate(self.spaces)}
        known = [sid for sid in ids if sid in order]
        known.sort(key=lambda sid: order[sid])
        return [space for space in self.spaces if space.id in known]

    def assignment(self, element: Element, spaces: list[SpaceElement]) -> str:
        """RoomAssignment value: ok, ambiguous, failed or not applicable."""
        if isinstance(element, SpaceElement):
            if element.id not in self.ctx.geometry.spaces:
                return "failed"
            return "ambiguous" if element.id in self.ambiguous else "ok"
        if element.kind not in (
            ElementKind.WALL,
            ElementKind.SLAB,
            ElementKind.CEILING,
            ElementKind.OPENING,
            ElementKind.WINDOW,
            ElementKind.DOOR,
        ):
            return NOT_APPLICABLE
        if not self.spaces:
            return NOT_APPLICABLE
        if not spaces:
            return "failed"
        if any(space.id in self.ambiguous for space in spaces):
            return "ambiguous"
        return "ok"


def _is_ambiguous(geometry: SpaceGeometry) -> bool:
    return geometry.matched_by == "order" or any(
        "ambiguous" in issue or "also claimed" in issue for issue in geometry.issues
    )


# -------------------------------------------------------------------------- standard property sets


def _known_function(element: WallElement) -> bool | None:
    function = element.params.function
    if function == WallFunction.EXTERIOR:
        return True
    if function == WallFunction.INTERIOR:
        return False
    return None


def _host_wall(ctx: ExportContext, element: Element) -> WallElement | None:
    if isinstance(element, OpeningElement):
        host = ctx.scene.elements.get(element.params.host_wall_id)
    elif isinstance(element, WindowElement | DoorElement):
        opening = ctx.scene.elements.get(element.params.opening_id)
        host = (
            ctx.scene.elements.get(opening.params.host_wall_id)
            if isinstance(opening, OpeningElement)
            else None
        )
    else:
        return None
    return host if isinstance(host, WallElement) else None


def _reference(element: Element) -> str:
    return element.tag or element.name or element.id


def standard_sets(
    ctx: ExportContext, element: Element
) -> tuple[dict[str, dict[str, PropValue]], dict[str, dict[str, float]]]:
    """Standard Pset and Qto contents of an element: ({pset: props}, {qto: quantities})."""
    quantities = ctx.geometry.quantities_of(element.id)
    psets: dict[str, dict[str, PropValue]] = {}
    qtos: dict[str, dict[str, float]] = {}

    if isinstance(element, WallElement):
        props: dict[str, PropValue] = {"Reference": _reference(element)}
        external = _known_function(element)
        if external is None:
            ctx.flag(element.id, "wall function unknown: IsExternal not written")
        else:
            props["IsExternal"] = external
        if element.params.load_bearing != Tristate.UNKNOWN:
            props["LoadBearing"] = element.params.load_bearing == Tristate.TRUE
        psets["Pset_WallCommon"] = props
        qtos["Qto_WallBaseQuantities"] = {
            name: quantities[name]
            for name in (
                "Length",
                "Width",
                "Height",
                "GrossSideArea",
                "NetSideArea",
                "GrossVolume",
                "NetVolume",
                "GrossFootprintArea",
            )
            if name in quantities
        }
    elif isinstance(element, SlabElement | CeilingElement):
        covering = (
            isinstance(element, CeilingElement)
            and ctx.scene.settings.ceiling_ifc_class == CeilingIfcClass.COVERING
        )
        if covering:
            psets["Pset_CoveringCommon"] = {"Reference": _reference(element)}
            qtos["Qto_CoveringBaseQuantities"] = {
                "Width": quantities["Width"],
                "GrossArea": quantities["GrossArea"],
                "NetArea": quantities["NetArea"],
            }
        else:
            slab_props: dict[str, PropValue] = {"Reference": _reference(element)}
            if isinstance(element, CeilingElement):
                tilt = _pitch_angle(ctx, element)
                if tilt is not None:
                    slab_props["PitchAngle"] = tilt
            psets["Pset_SlabCommon"] = slab_props
            qtos["Qto_SlabBaseQuantities"] = {
                "Depth": quantities["Width"],
                "Perimeter": quantities["Perimeter"],
                "GrossArea": quantities["GrossArea"],
                "NetArea": quantities["NetArea"],
                "GrossVolume": quantities["GrossVolume"],
                "NetVolume": quantities["NetVolume"],
            }
    elif isinstance(element, OpeningElement):
        psets["Pset_OpeningElementCommon"] = {"Reference": _reference(element)}
        opening = ctx.geometry.openings.get(element.id)
        qtos["Qto_OpeningElementBaseQuantities"] = {
            "Width": quantities["Width"],
            "Height": quantities["Height"],
            "Area": quantities["Area"],
        }
        if opening is not None:
            qtos["Qto_OpeningElementBaseQuantities"]["Depth"] = opening.depth
    elif isinstance(element, WindowElement | DoorElement):
        is_door = isinstance(element, DoorElement)
        common: dict[str, PropValue] = {"Reference": _reference(element)}
        host = _host_wall(ctx, element)
        external = _known_function(host) if host is not None else None
        if external is not None:
            common["IsExternal"] = external
        psets["Pset_DoorCommon" if is_door else "Pset_WindowCommon"] = common
        qtos["Qto_DoorBaseQuantities" if is_door else "Qto_WindowBaseQuantities"] = {
            "Width": quantities["Width"],
            "Height": quantities["Height"],
            "Area": quantities["Area"],
        }
    elif isinstance(element, StairElement):
        psets["Pset_StairCommon"] = {
            "Reference": _reference(element),
            "NumberOfRiser": ctx.file.create_entity(
                "IfcCountMeasure", int(element.params.step_count)
            ),
            "NumberOfTreads": ctx.file.create_entity(
                "IfcCountMeasure", int(quantities["NumberOfTreads"])
            ),
            "RiserHeight": ctx.file.create_entity(
                "IfcPositiveLengthMeasure", float(element.params.riser_height)
            ),
            "TreadLength": ctx.file.create_entity(
                "IfcPositiveLengthMeasure", float(element.params.tread_length)
            ),
        }
    elif isinstance(element, SpaceElement):
        psets["Pset_SpaceCommon"] = {
            "Reference": element.params.number or element.params.name or element.id,
            "IsExternal": False,
        }
        qtos["Qto_SpaceBaseQuantities"] = dict(quantities)
    return psets, {name: values for name, values in qtos.items() if values}


def _pitch_angle(ctx: ExportContext, element: CeilingElement) -> float | None:
    geometry = ctx.geometry.ceilings.get(element.id)
    if geometry is None or geometry.plane_normal is None:
        return None
    nx, ny, nz = geometry.plane_normal
    return math.acos(max(-1.0, min(1.0, nz / math.sqrt(nx * nx + ny * ny + nz * nz))))


def _user_value(value: PropertyValue) -> PropValue:
    return value.value


def _is_standard_pset(ifc_class: str, predefined: str, name: str) -> bool:
    if not name.startswith("Pset_"):
        return False
    return name in _PSET_TEMPLATES.get_applicable_names(ifc_class, predefined, pset_only=True)


def merge_user_properties(
    element: Element,
    product: Entity,
    psets: dict[str, dict[str, PropValue]],
) -> dict[str, PropValue]:
    """Merge user properties into standard psets; the rest is returned for Raumlot_User."""
    custom: dict[str, PropValue] = {}
    predefined = str(getattr(product, "PredefinedType", None) or "")
    for name, value in element.properties.items():
        target = value.pset
        if target is not None and _is_standard_pset(class_of(product), predefined, target):
            psets.setdefault(target, {})[name] = _user_value(value)
        else:
            custom[name] = _user_value(value)
    return custom


# ----------------------------------------------------------------------------- Raumlot psets


def _quality_properties(
    ctx: ExportContext, element: Element, quantities: dict[str, float]
) -> dict[str, PropValue]:
    scene_settings = ctx.scene.settings
    props: dict[str, PropValue] = {
        "ElementId": element.id,
        "Provenance": element.provenance.value,
        "RunId": ctx.run_id or NOT_APPLICABLE,
        "AiBackend": ctx.ai_backend or NOT_APPLICABLE,
        "AiModelId": ctx.ai_model or NOT_APPLICABLE,
        "ToleranceProfile": scene_settings.tolerance_profile,
        "DataSource": scene_settings.data_source.value,
        "Locked": element.locked,
        "Status": element.status.value,
    }
    metrics = element.metrics
    for name, value in (
        ("DeviationRms", metrics.deviation_rms),
        ("DeviationP95", metrics.deviation_p95),
        ("DeviationMax", metrics.deviation_max),
    ):
        if value is not None:
            props[name] = _length(ctx, value)
    if metrics.coverage is not None:
        props["Coverage"] = float(metrics.coverage)
    if metrics.confidence is not None:
        props["Confidence"] = float(metrics.confidence)
    if metrics.point_count is not None:
        props["PointCount"] = int(metrics.point_count)

    origins = {name: origin.value for name, origin in element.param_origin.items()}
    if origins:
        props["ParameterOrigins"] = _text(
            ctx, "; ".join(f"{name}={origin}" for name, origin in origins.items())
        )
        for name, origin_value in origins.items():
            props[f"Origin_{name}"] = origin_value
    for name, user_value in element.properties.items():
        props[f"PropertyOrigin_{name}"] = user_value.origin.value

    # Values without a standard quantity (MASTERPROMPT 10.1: Raumlot_Quality).
    if "SillHeight" in quantities:
        props["SillHeight"] = _length(ctx, quantities["SillHeight"])
    if isinstance(element, OpeningElement):
        props["RevealDepth"] = _length(ctx, element.params.reveal_depth)
        props["InsideSide"] = element.params.inside_side.value
    elif isinstance(element, WindowElement):
        props["Operation"] = element.params.operation.value
        props["SillProjection"] = _length(ctx, element.params.sill_projection)
        props["SillThickness"] = _length(ctx, element.params.sill_thickness)
    elif isinstance(element, DoorElement):
        props["Operation"] = element.params.operation.value
        props["SwingSide"] = element.params.swing_side.value
    elif isinstance(element, WallElement):
        props["Justification"] = element.params.justification.value
        props["BaseOffset"] = _length(ctx, element.params.base_offset)
    elif isinstance(element, SlabElement):
        props["TopElevation"] = _length(ctx, element.params.top_elevation)
    elif isinstance(element, CeilingElement):
        props["Elevation"] = _length(ctx, element.params.elevation)
    elif isinstance(element, StairElement):
        # No standard quantity for the width (MASTERPROMPT 10.1): Raumlot_Quality.
        props["StairWidth"] = _length(ctx, quantities["Width"])
        props["TotalRise"] = _length(ctx, quantities["TotalRise"])
        props["TotalRun"] = _length(ctx, quantities["TotalRun"])
        props["TotalLength"] = _length(ctx, quantities["Length"])
        props["FlightCount"] = len(element.params.flights)
        props["BaseOffset"] = _length(ctx, element.params.base_offset)
    if element.non_parametric:
        props["NonParametric"] = True
    if element.notes:
        props["Notes"] = _text(ctx, "; ".join(element.notes))
    if element.ai_comments:
        props["AiComments"] = _text(ctx, "; ".join(element.ai_comments))
    if element.source_instance_ids:
        props["SourceInstanceIds"] = _text(ctx, ", ".join(element.source_instance_ids))

    flags = ctx.flagged.get(element.id, [])
    props["Incomplete"] = bool(flags)
    if flags:
        props["MissingAttributes"] = _text(ctx, "; ".join(flags))
    return props


def _location_properties(
    ctx: ExportContext, element: Element, relations: SpaceRelations
) -> dict[str, PropValue]:
    scene = ctx.scene
    site = scene.sites.get(element.site_id or "")
    building = scene.buildings.get(element.building_id or "")
    storey = scene.storeys.get(element.storey_id or "")
    if site is None and element.site_id is None and scene.sites:
        site = next(iter(scene.sites.values()))
    if building is None and element.building_id is None:
        if storey is not None and storey.building_id in scene.buildings:
            building = scene.buildings[storey.building_id or ""]
        elif scene.buildings:
            building = next(iter(scene.buildings.values()))
    spaces = relations.of(element)
    label = crs_label(ctx)
    crs = scene.settings.crs
    props: dict[str, PropValue] = {
        "SiteName": (site.name or "Site") if site else NOT_APPLICABLE,
        "BuildingName": (building.name or "Building") if building else NOT_APPLICABLE,
        "StoreyName": (storey.name or "Storey") if storey else NOT_APPLICABLE,
        "StoreyElevation": _length(ctx, storey.elevation) if storey else NOT_APPLICABLE,
        "SpaceNames": ", ".join(s.params.name or s.name for s in spaces) or NOT_APPLICABLE,
        "SpaceNumbers": ", ".join(s.params.number for s in spaces if s.params.number)
        or NOT_APPLICABLE,
        "RoomAssignment": relations.assignment(element, spaces),
        "CrsName": label if label else UNREFERENCED,
        "HeightSystem": (
            (crs.vertical_datum if crs and crs.vertical_datum else "unspecified")
            if label
            else UNREFERENCED
        ),
    }
    bbox = element_bbox(ctx, element)
    offset = scene.settings.origin_offset
    names = ("X", "Y", "Z")
    if bbox is None:
        for axis in names:
            props[f"Reference{axis}"] = NOT_APPLICABLE
            props[f"BBoxMin{axis}"] = NOT_APPLICABLE
            props[f"BBoxMax{axis}"] = NOT_APPLICABLE
        return props
    low, high = bbox
    reference = (
        (low[0] + high[0]) / 2.0 + offset[0],
        (low[1] + high[1]) / 2.0 + offset[1],
        low[2] + offset[2],
    )
    for index, axis in enumerate(names):
        props[f"Reference{axis}"] = _length(ctx, reference[index])
        props[f"BBoxMin{axis}"] = _length(ctx, low[index] + offset[index])
        props[f"BBoxMax{axis}"] = _length(ctx, high[index] + offset[index])
    return props


def write_properties(ctx: ExportContext) -> None:
    """Standard psets and quantities, user properties and the Raumlot psets for every element."""
    relations = SpaceRelations(ctx)
    for element in ctx.scene.elements.values():
        product = ctx.entities.get(element.id)
        if product is None:
            continue
        psets, qtos = standard_sets(ctx, element)
        custom = merge_user_properties(element, product, psets)
        for name, props in psets.items():
            _add_pset(ctx, product, name, props)
        for name, values in qtos.items():
            _add_qto(ctx, product, name, values)
        if custom:
            _add_pset(ctx, product, CUSTOM_USER, custom)
        spaces = relations.of(element)
        assignment = relations.assignment(element, spaces)
        if assignment in ("ambiguous", "failed"):
            ctx.flag(element.id, f"room assignment {assignment}")
        quantities = ctx.geometry.quantities_of(element.id)
        _add_pset(ctx, product, "Raumlot_Quality", _quality_properties(ctx, element, quantities))
        _add_pset(ctx, product, "Raumlot_Location", _location_properties(ctx, element, relations))
