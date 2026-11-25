"""SAMcloud additions to the IFC writer: furniture / equipment and the project frame.

Furniture comes from the scene as ``FurnitureElement`` with free parameters (``params`` allows
extra fields).  Required: ``footprint`` (closed ring in metres, project frame, first point not
repeated), ``z_bottom`` and ``z_top`` (absolute metres).  Optional: ``object_type``, ``colour``,
``label``, ``semantic_label``, ``instance_id``, ``confidence``, ``z_source``, ``layer``,
``catalogue_kind``.  ``ifc_class`` / ``predefined_type`` of the element give the IFC class.

Per element:

* Body of a reconstructed item (``solids`` / ``mesh`` from samcloud.bim.furniture_recon): one
  ``IfcExtrudedAreaSolid`` per part with its own position (cylinders with an ``IfcCircleProfileDef``,
  SweptSolid) or one ``IfcTriangulatedFaceSet`` (Tessellation).
* Body otherwise: ``IfcExtrudedAreaSolid`` of the footprint from ``z_bottom`` to ``z_top`` (SweptSolid),
  an ``IfcBoundingBox`` ("Box") and the outline as ``IfcGeometricCurveSet`` in the FootPrint
  context, plus one surface style per class colour.
* One type object per ``(ifc_class, predefined type, object type)`` with ``Pset_*TypeCommon``
  (Reference) where the standard defines one.
* Property set ``SAMcloud_Scan`` (never ``Pset_``): InstanceId, Label, SemanticLabel,
  Confidence, ZSource, Layer, CatalogueKind, Length, Width, Height, ZMin; ``SAMcloud_Evidence = "manual"`` only
  for items the AI assistant added without point-cloud evidence (``evidence`` param).
* Containment in the storey; ``IfcRelReferencedInSpatialStructure`` to the room
  (``element.space_ids[0]``).

GlobalIds derive from the scene element id and stable type keys (deterministic files).
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any

import ifcopenshell
import ifcopenshell.api.pset
import ifcopenshell.api.root
import ifcopenshell.api.spatial
import ifcopenshell.api.style
import ifcopenshell.api.type
import ifcopenshell.util.pset
from shapely.geometry import Polygon

from samcloud.bim.core.ifc import shapes
from samcloud.bim.core.ifc.context import ExportContext
from samcloud.bim.core.ifc.ids import set_guid, set_key_guid
from samcloud.bim.core.ifc.spatial import container_of
from samcloud.bim.core.model import FurnitureElement

Entity = ifcopenshell.entity_instance

SCAN_PSET = "SAMcloud_Scan"
FRAME_PSET = "SAMcloud_Frame"
MIN_BODY_HEIGHT = 0.01
_TEMPLATES = ifcopenshell.util.pset.PsetQto("IFC4")


def _extra(element: FurnitureElement) -> dict[str, Any]:
    return dict(element.params.model_extra or {})


def _footprint_context(ctx: ExportContext) -> Entity | None:
    for context in ctx.file.by_type("IfcGeometricRepresentationSubContext"):
        if context.ContextIdentifier == "FootPrint":
            return context
    return None


def _style(ctx: ExportContext, colour: tuple[float, float, float]) -> Entity:
    key = ("style", round(colour[0], 3), round(colour[1], 3), round(colour[2], 3))
    if key in ctx.types:
        return ctx.types[key]
    style = ifcopenshell.api.style.add_style(ctx.file, name="SAMcloud " + "-".join(f"{c:.2f}" for c in colour))
    ifcopenshell.api.style.add_surface_style(
        ctx.file,
        style=style,
        ifc_class="IfcSurfaceStyleShading",
        attributes={
            "SurfaceColour": {"Name": None, "Red": colour[0], "Green": colour[1], "Blue": colour[2]},
            "Transparency": 0.0,
        },
    )
    ctx.types[key] = style
    return style


def _type_name(ifc_class: str, predefined: str, object_type: str) -> str:
    if object_type:
        return object_type
    return predefined.title() if predefined not in ("NOTDEFINED", "") else ifc_class[3:]


def _furniture_type(
    ctx: ExportContext, ifc_class: str, predefined: str, object_type: str
) -> Entity:
    key: tuple[object, ...] = ("furniture-type", ifc_class, predefined, object_type)
    if key in ctx.types:
        return ctx.types[key]
    name = _type_name(ifc_class, predefined, object_type)
    entity = ifcopenshell.api.root.create_entity(
        ctx.file, ifc_class=f"{ifc_class}Type", predefined_type=predefined, name=name
    )
    if predefined == "USERDEFINED":
        entity.ElementType = object_type or None
    if ifc_class == "IfcFurniture":
        entity.AssemblyPlace = "NOTDEFINED"
    set_key_guid(ctx, entity, f"type/{key!r}")
    template_name = f"Pset_{ifc_class[3:]}TypeCommon"
    if _TEMPLATES.get_by_name(template_name) is not None:
        pset = ifcopenshell.api.pset.add_pset(ctx.file, product=entity, name=template_name)
        ifcopenshell.api.pset.edit_pset(ctx.file, pset=pset, properties={"Reference": name})
    ctx.types[key] = entity
    return entity


def _clean_ring(points: list[list[float]]) -> list[tuple[float, float]]:
    ring = [(float(p[0]), float(p[1])) for p in points]
    if len(ring) > 1 and ring[0] == ring[-1]:
        ring = ring[:-1]
    twice_area = sum(
        ring[i][0] * ring[(i + 1) % len(ring)][1] - ring[(i + 1) % len(ring)][0] * ring[i][1]
        for i in range(len(ring))
    )
    if twice_area < 0:  # counter-clockwise
        ring.reverse()
    return ring


@dataclass(frozen=True, slots=True)
class _Reconstructed:
    """Validated reconstructed shape of one element (scene metres, absolute)."""

    solids: list[dict[str, Any]]
    vertices: list[tuple[float, float, float]]
    faces: list[tuple[int, int, int]]
    bounds: tuple[float, float, float, float]
    z0: float
    z1: float


def _reconstructed_geometry(extra: dict[str, Any]) -> _Reconstructed | None:
    """The element's ``solids`` (vertical prisms) or ``mesh`` (closed triangles) when usable, else None (footprint body)."""
    mesh = extra.get("mesh")
    if isinstance(mesh, dict) and mesh.get("vertices") and mesh.get("faces"):
        vertices = [(float(v[0]), float(v[1]), float(v[2])) for v in mesh["vertices"]]
        count = len(vertices)
        faces = [
            (int(a), int(b), int(c)) for a, b, c in mesh["faces"]
            if 0 <= int(a) < count and 0 <= int(b) < count and 0 <= int(c) < count and len({int(a), int(b), int(c)}) == 3
        ]
        if count >= 4 and len(faces) >= 4:
            xs, ys, zs = [v[0] for v in vertices], [v[1] for v in vertices], [v[2] for v in vertices]
            if max(zs) - min(zs) >= MIN_BODY_HEIGHT:
                return _Reconstructed([], vertices, faces, (min(xs), min(ys), max(xs), max(ys)), min(zs), max(zs))
    usable: list[dict[str, Any]] = []
    for solid in extra.get("solids") or []:
        if not isinstance(solid, dict):
            continue
        ring = _clean_ring(solid.get("outline") or [])
        bottom, top = float(solid.get("z_bottom", 0.0)), float(solid.get("z_top", 0.0))
        if top - bottom < 0.005:
            continue
        if solid.get("type") == "cylinder" and solid.get("center") is not None and float(solid.get("radius") or 0.0) > 1e-3:
            cx, cy = float(solid["center"][0]), float(solid["center"][1])
            r = float(solid["radius"])
            usable.append({"type": "cylinder", "center": (cx, cy), "radius": r, "z_bottom": bottom, "z_top": top,
                           "ring": [(cx - r, cy - r), (cx + r, cy - r), (cx + r, cy + r), (cx - r, cy + r)]})
            continue
        if len(ring) >= 3 and Polygon(ring).is_valid and Polygon(ring).area > 1e-6:
            usable.append({"type": "prism", "ring": ring, "z_bottom": bottom, "z_top": top})
    if not usable:
        return None
    xs = [p[0] for solid in usable for p in solid["ring"]]
    ys = [p[1] for solid in usable for p in solid["ring"]]
    return _Reconstructed(
        usable, [], [], (min(xs), min(ys), max(xs), max(ys)),
        min(solid["z_bottom"] for solid in usable), max(solid["z_top"] for solid in usable),
    )


def _reconstructed_body(ctx: ExportContext, recon: _Reconstructed, origin: tuple[float, float, float]) -> Entity:
    """Body representation: one extruded solid per part (SweptSolid; cylinders with a circle profile) or one
    IfcTriangulatedFaceSet (Tessellation)."""
    f = ctx.file
    ox, oy, oz = origin
    if recon.vertices:
        local = [(x - ox, y - oy, z - oz) for x, y, z in recon.vertices]
        return shapes.tessellation_representation(ctx, shapes.triangulated_face_set(f, local, recon.faces))
    items: list[Entity] = []
    for solid in recon.solids:
        depth = solid["z_top"] - solid["z_bottom"]
        start = (0.0, 0.0, solid["z_bottom"] - oz)
        if solid["type"] == "cylinder":
            cx, cy = solid["center"]
            profile = f.createIfcCircleProfileDef(
                "AREA", None, f.createIfcAxis2Placement2D(f.createIfcCartesianPoint((cx - ox, cy - oy)), None), float(solid["radius"])
            )
        else:
            profile = shapes.polygon_profile(f, [(x - ox, y - oy) for x, y in solid["ring"]])
        items.append(shapes.extrusion(f, profile, depth, origin=start))
    return shapes.swept_solid_representation(ctx, items)


def write_furniture(ctx: ExportContext) -> int:
    """Write all ``FurnitureElement`` items that have a footprint; return how many were written."""
    f = ctx.file
    footprint_context = _footprint_context(ctx)
    written = 0
    for element in ctx.scene.elements.values():
        if not isinstance(element, FurnitureElement):
            continue
        extra = _extra(element)
        raw = extra.get("footprint")
        if not raw or len(raw) < 3 or extra.get("z_bottom") is None or extra.get("z_top") is None:
            ctx.flag(element.id, "furniture without footprint or heights: not exported")
            continue
        ring = _clean_ring(raw)
        polygon = Polygon(ring)
        if len(ring) < 3 or not polygon.is_valid or polygon.area <= 1e-6:
            ctx.flag(element.id, "furniture footprint is not a valid polygon: not exported")
            continue
        z0, z1 = float(extra["z_bottom"]), float(extra["z_top"])
        if z1 - z0 < MIN_BODY_HEIGHT:
            ctx.flag(element.id, "furniture height below 1 cm: not exported")
            continue
        solid_parts: list[tuple[list[tuple[float, float]], Polygon, float, float]] = []
        for part in extra.get("parts") or [{"footprint": raw, "z_bottom": z0, "z_top": z1}]:
            if not isinstance(part, dict):
                continue
            part_ring = _clean_ring(part.get("footprint") or [])
            part_polygon = Polygon(part_ring)
            bottom = float(part.get("z_bottom", z0))
            top = float(part.get("z_top", z1))
            if len(part_ring) >= 3 and part_polygon.is_valid and part_polygon.area > 1e-6 and top - bottom >= MIN_BODY_HEIGHT:
                solid_parts.append((part_ring, part_polygon, bottom, top))
        if not solid_parts:
            ctx.flag(element.id, "furniture has no valid solid parts: not exported")
            continue
        z0 = min(part[2] for part in solid_parts)
        z1 = max(part[3] for part in solid_parts)
        recon = _reconstructed_geometry(extra)
        if recon is not None:
            z0, z1 = recon.z0, recon.z1

        ifc_class = element.ifc_class or "IfcFurniture"
        predefined = element.predefined_type or "NOTDEFINED"
        object_type = str(extra.get("object_type") or "")
        product = ifcopenshell.api.root.create_entity(
            f, ifc_class=ifc_class, name=element.name or _type_name(ifc_class, predefined, object_type)
        )
        set_guid(ctx, product, element.id)
        product.Tag = element.id
        container = container_of(ctx, element.storey_id, element.building_id)
        ifcopenshell.api.spatial.assign_container(f, products=[product], relating_structure=container)

        type_entity = _furniture_type(ctx, ifc_class, predefined, object_type)
        ifcopenshell.api.type.assign_type(
            f, related_objects=[product], relating_type=type_entity, should_map_representations=False
        )
        product.PredefinedType = predefined
        product.ObjectType = object_type or _type_name(ifc_class, predefined, object_type)

        if recon is not None:
            min_x, min_y, max_x, max_y = recon.bounds
        else:
            min_x = min(point[0] for part in solid_parts for point in part[0])
            min_y = min(point[1] for part in solid_parts for point in part[0])
            max_x = max(point[0] for part in solid_parts for point in part[0])
            max_y = max(point[1] for part in solid_parts for point in part[0])
        shapes.place(ctx, product, (min_x, min_y, z0))
        local = [(x - min_x, y - min_y) for x, y in ring]
        if recon is not None:
            body = _reconstructed_body(ctx, recon, (min_x, min_y, z0))
        else:
            solids = [
                shapes.extrusion(
                    f, shapes.polygon_profile(f, [(x - min_x, y - min_y) for x, y in part_ring]), top - bottom,
                    origin=(0.0, 0.0, bottom - z0),
                )
                for part_ring, _part_polygon, bottom, top in solid_parts
            ]
            body = shapes.swept_solid_representation(ctx, solids)
        colour = extra.get("colour")
        if colour:
            ifcopenshell.api.style.assign_representation_styles(
                f, shape_representation=body, styles=[_style(ctx, (colour[0], colour[1], colour[2]))]
            )
        representations = [body]
        box = f.createIfcBoundingBox(
            f.createIfcCartesianPoint((0.0, 0.0, 0.0)), max_x - min_x, max_y - min_y, z1 - z0
        )
        representations.append(f.createIfcShapeRepresentation(ctx.body, "Box", "BoundingBox", [box]))
        if footprint_context is not None:
            points = [f.createIfcCartesianPoint((float(x), float(y))) for x, y in local]
            points.append(points[0])
            curve_set = f.createIfcGeometricCurveSet([f.createIfcPolyline(points)])
            representations.append(
                f.createIfcShapeRepresentation(footprint_context, "FootPrint", "GeometricCurveSet", [curve_set])
            )
        shapes.assign_representations(ctx, product, representations)

        rectangle = polygon.minimum_rotated_rectangle
        corners = list(rectangle.exterior.coords)
        edges = sorted(
            math.hypot(corners[i + 1][0] - corners[i][0], corners[i + 1][1] - corners[i][1]) for i in range(2)
        )
        props: dict[str, Any] = {
            "Label": str(extra.get("label") or ""),
            "SemanticLabel": str(extra.get("semantic_label") or ""),
            "ZSource": str(extra.get("z_source") or "default"),
            "Layer": str(extra.get("layer") or ""),
            "CatalogueKind": str(extra.get("catalogue_kind") or ""),
            "Length": f.create_entity("IfcLengthMeasure", float(edges[1])),
            "Width": f.create_entity("IfcLengthMeasure", float(edges[0])),
            "Height": f.create_entity("IfcLengthMeasure", float(z1 - z0)),
            "ZMin": f.create_entity("IfcLengthMeasure", float(z0)),
            "PartCount": len(recon.solids) if recon is not None and recon.solids else len(solid_parts),
            "ShapeSource": str(extra.get("shape_source") or "footprint") if recon is not None else "footprint",
        }
        if extra.get("instance_id") is not None:
            props["InstanceId"] = int(extra["instance_id"])
        if extra.get("confidence") is not None:
            props["Confidence"] = float(extra["confidence"])
        if extra.get("facing_deg") is not None:
            props["FacingDeg"] = float(extra["facing_deg"])
        if extra.get("back_wall_id"):
            props["BackWallId"] = str(extra["back_wall_id"])
        if extra.get("evidence") == "manual":  # added by the AI assistant without point-cloud evidence
            props["SAMcloud_Evidence"] = "manual"
        product.Description = element.description or (
            f"{_type_name(ifc_class, predefined, object_type)}, "
            f"{edges[1]:.2f} x {edges[0]:.2f} x {z1 - z0:.2f} m"
        )
        pset = ifcopenshell.api.pset.add_pset(f, product=product, name=SCAN_PSET)
        ifcopenshell.api.pset.edit_pset(f, pset=pset, properties=props)

        for space_id in element.space_ids:
            space = ctx.entities.get(space_id)
            if space is not None:
                ifcopenshell.api.spatial.reference_structure(f, products=[product], relating_structure=space)
                break
        written += 1
    return written


def write_project_frame(ctx: ExportContext) -> None:
    """Property set ``SAMcloud_Frame`` on the IfcProject (source frame for later georeferencing)."""
    frame = ctx.scene.derived.get("samcloud_frame")
    if not isinstance(frame, dict) or ctx.project is None:
        return
    props: dict[str, Any] = {}
    for key, value in frame.items():
        if isinstance(value, bool) or value is None:
            continue
        if isinstance(value, (int, float)):
            props[str(key)] = float(value)
        elif isinstance(value, str):
            props[str(key)] = value
        else:
            props[str(key)] = json.dumps(value, separators=(",", ":"))
    if props:
        pset = ifcopenshell.api.pset.add_pset(ctx.file, product=ctx.project, name=FRAME_PSET)
        ifcopenshell.api.pset.edit_pset(ctx.file, pset=pset, properties=props)
