"""IFC4 mapping for outdoor survey elements."""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Sequence

import ifcopenshell
import ifcopenshell.api.aggregate
import ifcopenshell.api.geometry
import ifcopenshell.api.pset
import ifcopenshell.api.root
import ifcopenshell.api.spatial
import numpy as np

from samcloud.bim.core.ifc import shapes
from samcloud.bim.core.ifc.context import ExportContext
from samcloud.bim.core.ifc.elements import element_name
from samcloud.bim.core.ifc.ids import set_guid
from samcloud.bim.core.model import (
    BuildingElement, CurbElement, FenceElement, ManholeElement, MarkingElement,
    PavementElement, ShrubElement, SiteWallElement, TerrainElement, TreeElement,
    VegetationAreaElement,
)

Entity = ifcopenshell.entity_instance
Vec3 = tuple[float, float, float]


def _site(ctx: ExportContext, element: object) -> Entity:
    site_id = getattr(element, "site_id", None)
    return ctx.sites.get(site_id or "", ctx.sites[ctx.default_site_id])


def _product(ctx: ExportContext, element: object, ifc_class: str, label: str) -> Entity:
    product = ifcopenshell.api.root.create_entity(ctx.file, ifc_class=ifc_class, name=element_name(element, label))
    set_guid(ctx, product, element.id)
    product.Description = element.description
    if hasattr(product, "Tag"):
        product.Tag = element.id
    ctx.entities[element.id] = product
    ifcopenshell.api.spatial.assign_container(ctx.file, products=[product], relating_structure=_site(ctx, element))
    shapes.place(ctx, product, (0.0, 0.0, 0.0))
    return product


def _assign_mesh(ctx: ExportContext, product: Entity, vertices: Sequence[Vec3], triangles: Sequence[tuple[int, int, int]]) -> None:
    item = shapes.triangulated_face_set(ctx.file, vertices, triangles)
    shapes.assign_representations(ctx, product, [shapes.tessellation_representation(ctx, item)])


def _footprint_representation(
    ctx: ExportContext, outline: Sequence[tuple[float, float]], base_z: float
) -> Entity:
    """Create a closed plan footprint at its measured terrain elevation."""
    context = next(
        (
            item
            for item in ctx.file.by_type("IfcGeometricRepresentationSubContext")
            if item.ContextIdentifier == "FootPrint"
        ),
        None,
    )
    if context is None:
        raise ValueError("IFC FootPrint representation context is missing")
    points = [(x, y, base_z) for x, y in outline]
    curve = ctx.file.createIfcPolyline(
        [ctx.file.createIfcCartesianPoint(point) for point in [*points, points[0]]]
    )
    return ctx.file.createIfcShapeRepresentation(
        context, "FootPrint", "GeometricCurveSet", [ctx.file.createIfcGeometricCurveSet([curve])]
    )


def _prism_mesh(outline: Sequence[tuple[float, float]], base: float, height: float) -> tuple[list[Vec3], list[tuple[int, int, int]]]:
    count = len(outline)
    vertices = [(x, y, base) for x, y in outline] + [(x, y, base + height) for x, y in outline]
    triangles: list[tuple[int, int, int]] = []
    for index in range(1, count - 1):
        triangles.extend(((0, index + 1, index), (count, count + index, count + index + 1)))
    for index in range(count):
        following = (index + 1) % count
        triangles.extend(((index, following, count + following), (index, count + following, count + index)))
    return vertices, triangles


def _pset(ctx: ExportContext, product: Entity, category: str, element: object, values: dict[str, object]) -> None:
    """Write measured values plus origin, status and evidence in one category Pset."""
    props: dict[str, object] = dict(values)
    for name, origin in element.param_origin.items():
        props[f"Origin_{name}"] = origin.value
    props["Status"] = element.status.value
    if element.source_instance_ids:
        props["EvidenceViews"] = ", ".join(element.source_instance_ids)
    if element.metrics.point_count is not None:
        props["EvidencePoints"] = element.metrics.point_count
    if element.metrics.deviation_rms is not None:
        props["EvidenceRmse"] = element.metrics.deviation_rms
    for name, value in element.evidence.items():
        if isinstance(value, (str, int, float, bool)):
            props[f"Evidence_{name}"] = value
        elif isinstance(value, (list, tuple)):
            props[f"Evidence_{name}"] = ", ".join(str(item) for item in value)
        else:
            props[f"Evidence_{name}"] = str(value)
    pset = ifcopenshell.api.pset.add_pset(ctx.file, product=product, name=f"SAMcloud_{category}")
    ifcopenshell.api.pset.edit_pset(ctx.file, pset=pset, properties=props)


def _axis_footprint(axis: Sequence[tuple[float, float]], thickness: float) -> list[tuple[float, float]]:
    (x1, y1), (x2, y2) = axis[0], axis[-1]
    length = math.hypot(x2 - x1, y2 - y1)
    nx, ny = -(y2 - y1) / length * thickness / 2.0, (x2 - x1) / length * thickness / 2.0
    return [(x1 + nx, y1 + ny), (x2 + nx, y2 + ny), (x2 - nx, y2 - ny), (x1 - nx, y1 - ny)]


def _arc_is_continuous(axis: Sequence[Vec3]) -> bool:
    if len(axis) < 4:
        return False
    xy = np.asarray([(point[0], point[1]) for point in axis], dtype=float)
    a = np.column_stack((2.0 * xy[:, 0], 2.0 * xy[:, 1], np.ones(len(xy))))
    b = np.sum(xy * xy, axis=1)
    center_x, center_y, constant = np.linalg.lstsq(a, b, rcond=None)[0]
    radius = math.sqrt(max(0.0, constant + center_x * center_x + center_y * center_y))
    residual = np.abs(np.hypot(xy[:, 0] - center_x, xy[:, 1] - center_y) - radius)
    return radius < 50.0 and float(residual.max()) < 0.01


def _curb_extrusions(ctx: ExportContext, curb: CurbElement) -> list[Entity]:
    axis = [np.asarray(point, dtype=float) for point in curb.params.axis]
    solids: list[Entity] = []
    for index, (start, end) in enumerate(zip(axis, axis[1:])):
        vector = end - start
        length = float(np.linalg.norm(vector))
        if length <= 1e-9:
            continue
        direction = vector / length
        extension_start = extension_end = 0.0
        for joint, target in ((index, "start"), (index + 1, "end")):
            if joint <= 0 or joint >= len(axis) - 1:
                continue
            before = axis[joint] - axis[joint - 1]
            after = axis[joint + 1] - axis[joint]
            before[2] = after[2] = 0.0
            if np.linalg.norm(before) > 1e-9 and np.linalg.norm(after) > 1e-9:
                cosine = float(np.clip(np.dot(before, after) / (np.linalg.norm(before) * np.linalg.norm(after)), -1.0, 1.0))
                extension = curb.params.width / 2.0 * math.tan(math.acos(cosine) / 2.0)
                if target == "start":
                    extension_start = extension
                else:
                    extension_end = extension
        origin = start - direction * extension_start
        profile = shapes.rectangle_profile(ctx.file, (curb.params.height / 2.0, 0.0), curb.params.height, curb.params.width)
        solids.append(shapes.extrusion(ctx.file, profile, length + extension_start + extension_end, origin=tuple(origin), axis=tuple(direction), ref_direction=(0.0, 0.0, 1.0)))
    return solids


def _pavement_mesh(element: PavementElement) -> tuple[list[Vec3], list[tuple[int, int, int]]]:
    top = list(element.params.vertices)
    count = len(top)
    bottom = [(x, y, z - element.params.thickness) for x, y, z in top]
    triangles: list[tuple[int, int, int]] = []
    for a, b, c in element.params.triangles:
        triangles.extend(((a, b, c), (count + c, count + b, count + a)))
    edges: Counter[tuple[int, int]] = Counter()
    for a, b, c in element.params.triangles:
        for edge in ((a, b), (b, c), (c, a)):
            edges[tuple(sorted(edge))] += 1
    for a, b, c in element.params.triangles:
        for first, second in ((a, b), (b, c), (c, a)):
            if edges[tuple(sorted((first, second)))] == 1:
                triangles.extend(((first, second, count + second), (first, count + second, count + first)))
    return top + bottom, triangles


def write_outdoor(ctx: ExportContext) -> None:
    """Write site-contained outdoor survey products and LoD2 building envelopes."""
    for element in ctx.scene.elements.values():
        if isinstance(element, BuildingElement):
            building = ctx.buildings.get(element.id)
            if building is None:
                continue
            ctx.entities[element.id] = building
            walls = [triangle for triangle, role in zip(element.params.triangles, element.params.face_roles, strict=True) if role == 1]
            roofs = [triangle for triangle, role in zip(element.params.triangles, element.params.face_roles, strict=True) if role == 2]
            if walls:
                wall = ifcopenshell.api.root.create_entity(ctx.file, ifc_class="IfcWall", name=f"{element_name(element, 'Building')} exterior wall")
                wall.ObjectType = "EXTERIOR"
                wall.PredefinedType = "STANDARD"
                ifcopenshell.api.spatial.assign_container(ctx.file, products=[wall], relating_structure=building)
                shapes.place(ctx, wall, (0.0, 0.0, 0.0))
                _assign_mesh(ctx, wall, element.params.vertices, walls)
            if roofs:
                roof = ifcopenshell.api.root.create_entity(ctx.file, ifc_class="IfcRoof", name=f"{element_name(element, 'Building')} roof")
                roof.PredefinedType = "FLAT_ROOF"
                slab = ifcopenshell.api.root.create_entity(ctx.file, ifc_class="IfcSlab", name=f"{element_name(element, 'Building')} roof surface")
                slab.PredefinedType = "ROOF"
                ifcopenshell.api.aggregate.assign_object(ctx.file, products=[slab], relating_object=roof)
                ifcopenshell.api.spatial.assign_container(ctx.file, products=[roof], relating_structure=building)
                shapes.place(ctx, roof, (0.0, 0.0, 0.0))
                shapes.place(ctx, slab, (0.0, 0.0, 0.0))
                _assign_mesh(ctx, slab, element.params.vertices, roofs)
            _pset(ctx, building, "Building", element, {"TriangleCount": len(element.params.triangles)})
        elif isinstance(element, TerrainElement):
            product = _product(ctx, element, "IfcGeographicElement", "Terrain")
            product.PredefinedType = "TERRAIN"
            if element.params.triangles:
                _assign_mesh(ctx, product, element.params.vertices, element.params.triangles)
            _pset(ctx, product, "Terrain", element, {"TriangleCount": len(element.params.triangles)})
        elif isinstance(element, TreeElement):
            product = _product(ctx, element, "IfcGeographicElement", "Tree")
            product.PredefinedType, product.ObjectType = "USERDEFINED", "Baum"
            trunk, crown = shapes.stylised_tree(ctx.file, element.params.trunk_xy, element.params.base_z, element.params.height, element.params.crown_diameter, element.params.trunk_diameter)
            shapes.assign_representations(ctx, product, [shapes.swept_solid_representation(ctx, [trunk]), shapes.tessellation_representation(ctx, crown)])
            _pset(ctx, product, "Tree", element, {"Height": element.params.height, "CrownDiameter": element.params.crown_diameter, "TrunkDiameter": element.params.trunk_diameter})
        elif isinstance(element, ShrubElement):
            category, label, object_type = "Shrub", "Shrub", "Busch"
            product = _product(ctx, element, "IfcGeographicElement", label)
            product.PredefinedType, product.ObjectType = "USERDEFINED", object_type
            vertices, triangles = _prism_mesh(
                element.params.outline, element.params.base_z, element.params.height
            )
            _assign_mesh(ctx, product, vertices, triangles)
            _pset(
                ctx,
                product,
                category,
                element,
                {"BaseZ": element.params.base_z, "Height": element.params.height},
            )
        elif isinstance(element, VegetationAreaElement):
            product = _product(ctx, element, "IfcGeographicElement", "Vegetation area")
            product.PredefinedType, product.ObjectType = "USERDEFINED", "Vegetationsfläche"
            shapes.assign_representations(
                ctx,
                product,
                [_footprint_representation(ctx, element.params.outline, element.params.base_z)],
            )
            _pset(ctx, product, "VegetationArea", element, {"BaseZ": element.params.base_z})
        elif isinstance(element, SiteWallElement):
            product = _product(ctx, element, "IfcWall", "Site wall")
            product.ObjectType, product.PredefinedType = "Mauer", "STANDARD"
            outline = element.params.footprint or _axis_footprint(element.params.axis or [], element.params.thickness or 0.2)
            profile = shapes.polygon_profile(ctx.file, outline)
            solid = shapes.extrusion(ctx.file, profile, element.params.top_z - element.params.base_z, origin=(0.0, 0.0, element.params.base_z))
            shapes.assign_representations(ctx, product, [shapes.swept_solid_representation(ctx, [solid])])
            _pset(ctx, product, "SiteWall", element, {"BaseZ": element.params.base_z, "TopZ": element.params.top_z, "Thickness": element.params.thickness or 0.0})
        elif isinstance(element, FenceElement):
            product = _product(ctx, element, "IfcRailing", "Fence")
            product.ObjectType = "Zaun"
            solids = _curb_extrusions(ctx, CurbElement(params={"axis": element.params.axis, "width": 0.04, "height": element.params.height}))
            shapes.assign_representations(ctx, product, [shapes.swept_solid_representation(ctx, solids)])
            _pset(ctx, product, "Fence", element, {"Height": element.params.height})
        elif isinstance(element, PavementElement):
            product = _product(ctx, element, "IfcCivilElement", "Pavement")
            product.ObjectType = {"road": "Strasse", "sidewalk": "Trottoir", "path": "Weg"}[element.params.kind]
            vertices, triangles = _pavement_mesh(element)
            _assign_mesh(ctx, product, vertices, triangles)
            _pset(ctx, product, "Pavement", element, {"Thickness": element.params.thickness, "Kind": element.params.kind})
        elif isinstance(element, CurbElement):
            product = _product(ctx, element, "IfcCivilElement", "Curb")
            product.ObjectType = "Randstein"
            if element.params.curb_sweep == "auto" and _arc_is_continuous(element.params.axis):
                solid = shapes.swept_curb(ctx.file, element.params.axis, element.params.width, element.params.height)
                shapes.assign_representations(ctx, product, [shapes.swept_solid_representation(ctx, [solid])])
            else:
                shapes.assign_representations(ctx, product, [shapes.swept_solid_representation(ctx, _curb_extrusions(ctx, element))])
            _pset(ctx, product, "Curb", element, {"Width": element.params.width, "Height": element.params.height, "Sweep": element.params.curb_sweep})
        elif isinstance(element, MarkingElement):
            product = _product(ctx, element, "IfcAnnotation", "Marking")
            product.ObjectType = "Markierung"
            points = [tuple(point) if len(point) == 3 else (point[0], point[1], 0.0) for point in element.params.points]
            curve = ctx.file.createIfcPolyline([ctx.file.createIfcCartesianPoint(point) for point in points + ([points[0]] if element.params.closed else [])])
            representation = ctx.file.createIfcShapeRepresentation(ctx.body, "Body", "GeometricCurveSet", [ctx.file.createIfcGeometricCurveSet([curve])])
            shapes.assign_representations(ctx, product, [representation])
            _pset(ctx, product, "Marking", element, {"Closed": element.params.closed})
        elif isinstance(element, ManholeElement):
            product = _product(ctx, element, "IfcDistributionChamberElement", "Manhole")
            product.PredefinedType, product.ObjectType = "MANHOLE", element.params.type
            profile = shapes.polygon_profile(ctx.file, [(element.params.diameter / 2.0 * math.cos(2 * math.pi * i / 12), element.params.diameter / 2.0 * math.sin(2 * math.pi * i / 12)) for i in range(12)])
            solid = shapes.extrusion(ctx.file, profile, 0.05, origin=element.params.center)
            shapes.assign_representations(ctx, product, [shapes.swept_solid_representation(ctx, [solid])])
            _pset(ctx, product, "Manhole", element, {"Diameter": element.params.diameter, "Z_DECKEL": element.params.center[2], "Type": element.params.type})
