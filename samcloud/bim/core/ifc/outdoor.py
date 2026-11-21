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
import shapely
from shapely.geometry import Polygon

from samcloud.bim.core.ifc import shapes
from samcloud.bim.core.ifc.context import ExportContext
from samcloud.bim.core.ifc.elements import element_name
from samcloud.bim.core.ifc.ids import set_guid
from samcloud.outdoor_model.style import load_outdoor_style
from samcloud.bim.core.model import (
    BuildingElement, CurbElement, FenceElement, ManholeElement, MarkingElement,
    PavementElement, ShrubElement, SiteWallElement, TerrainElement, TreeElement,
    VegetationAreaElement, WallElement,
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


_PSET_SKIP_EVIDENCE = frozenset({"lawn_triangle_indices"})  # bulk triangle index list, not a property


def _submesh(
    vertices: np.ndarray, triangles: np.ndarray
) -> tuple[list[Vec3], list[tuple[int, int, int]]]:
    """Mesh of ``triangles`` with the used vertices only (re-indexed)."""
    used, inverse = np.unique(triangles.ravel(), return_inverse=True)
    return (
        [tuple(float(v) for v in vertices[i]) for i in used],
        [tuple(int(i) for i in row) for row in inverse.reshape(triangles.shape)],
    )


def _split_lawn_surfaces(
    ctx: ExportContext,
) -> tuple[dict[str, tuple[list[Vec3], list[tuple[int, int, int]]]], dict[str, tuple[list[Vec3], list[tuple[int, int, int]]]]]:
    """Hand the lawn-labelled terrain triangles over to the lawn elements.

    Returns ``(lawn_meshes, terrain_meshes)`` keyed by element id. The lawn triangles are the ones listed in
    the terrain evidence ``lawn_triangle_indices`` (all triangles when absent) whose centroid lies inside the
    outline of a vegetation area (smallest outline wins); they are removed from the terrain mesh, so terrain,
    lawn and pavements together cover the TIN without gap or overlap. Lawn-labelled triangles outside every
    outline and lawns without triangles stay as they are (terrain keeps them, the lawn keeps its footprint).
    """
    terrains = [e for e in ctx.scene.elements.values() if isinstance(e, TerrainElement)]
    lawns = [e for e in ctx.scene.elements.values() if isinstance(e, VegetationAreaElement)]
    if len(terrains) != 1 or not lawns or not terrains[0].params.triangles:
        return {}, {}
    terrain = terrains[0]
    vertices = np.asarray(terrain.params.vertices, dtype=float)
    triangles = np.asarray(terrain.params.triangles, dtype=np.int64)
    listed = terrain.evidence.get("lawn_triangle_indices")
    candidates = np.arange(len(triangles)) if listed is None else np.asarray(listed, dtype=np.int64).reshape(-1)
    candidates = candidates[(candidates >= 0) & (candidates < len(triangles))]
    if len(candidates) == 0:
        return {}, {}
    centroids = vertices[triangles[candidates]][:, :, :2].mean(axis=1)
    owner = np.full(len(candidates), -1, dtype=np.int64)
    polygons = []
    for index, lawn in enumerate(lawns):
        polygon = Polygon(lawn.params.outline)
        polygons.append(polygon if polygon.is_valid else polygon.buffer(0))
    for index in sorted(range(len(lawns)), key=lambda i: -polygons[i].area):  # larger first, smaller outlines overwrite
        owner[shapely.contains_xy(polygons[index], centroids[:, 0], centroids[:, 1])] = index
    taken = owner >= 0
    if not taken.any() or int(taken.sum()) >= len(triangles):
        return {}, {}
    lawn_meshes = {
        lawns[index].id: _submesh(vertices, triangles[candidates[owner == index]])
        for index in range(len(lawns))
        if (owner == index).any()
    }
    keep = np.ones(len(triangles), dtype=bool)
    keep[candidates[taken]] = False
    return lawn_meshes, {terrain.id: _submesh(vertices, triangles[keep])}


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
        if name in _PSET_SKIP_EVIDENCE:
            continue
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


def _write_facade_walls(ctx: ExportContext, building: BuildingElement) -> int:
    """Give each facade IfcWall of ``building`` the exact LoD2 wall faces of its facade plane.

    The generic wall writer extrudes a flat prism; here its body is replaced by the mesh faces listed in
    the wall's ``face_indices`` evidence, so walls end exactly at the LoD2 roof edge and terrain bottom.
    Returns the number of walls written.
    """
    count = 0
    triangles = list(building.params.triangles)
    for element in ctx.scene.elements.values():
        if not isinstance(element, WallElement) or element.building_id != building.id:
            continue
        indices = element.evidence.get("face_indices")
        wall = ctx.entities.get(element.id)
        if not indices or wall is None:
            continue
        used = sorted({int(v) for i in indices for v in triangles[int(i)]})
        remap = {old: new for new, old in enumerate(used)}
        sub_vertices = [building.params.vertices[old] for old in used]
        sub_triangles = [tuple(remap[int(v)] for v in triangles[int(i)]) for i in indices]
        if wall.Representation is not None:
            for representation in list(wall.Representation.Representations):
                ifcopenshell.api.geometry.unassign_representation(ctx.file, product=wall, representation=representation)
                ifcopenshell.api.geometry.remove_representation(ctx.file, representation=representation)
        shapes.place(ctx, wall, (0.0, 0.0, 0.0))
        _assign_mesh(ctx, wall, sub_vertices, sub_triangles)
        count += 1
    return count


def write_outdoor(ctx: ExportContext) -> None:
    """Write site-contained outdoor survey products and LoD2 building envelopes."""
    lawn_meshes, terrain_meshes = _split_lawn_surfaces(ctx)
    for element in ctx.scene.elements.values():
        if isinstance(element, BuildingElement):
            building = ctx.buildings.get(element.id)
            if building is None:
                continue
            ctx.entities[element.id] = building
            walls = [triangle for triangle, role in zip(element.params.triangles, element.params.face_roles, strict=True) if role == 1]
            roofs = [triangle for triangle, role in zip(element.params.triangles, element.params.face_roles, strict=True) if role == 2]
            facade_walls = _write_facade_walls(ctx, element)
            if walls and not facade_walls:
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
            terrain_vertices, terrain_triangles = terrain_meshes.get(element.id, (element.params.vertices, element.params.triangles))
            if terrain_triangles:
                _assign_mesh(ctx, product, terrain_vertices, terrain_triangles)
            _pset(ctx, product, "Terrain", element, {"TriangleCount": len(terrain_triangles)})
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
            product = _product(ctx, element, "IfcGeographicElement", "Rasen")
            product.PredefinedType, product.ObjectType = "USERDEFINED", "Vegetationsfläche"
            surface = lawn_meshes.get(element.id)
            if surface is not None:  # lawn surface: the lawn-labelled TIN triangles inside the outline
                _assign_mesh(ctx, product, *surface)
            else:
                shapes.assign_representations(
                    ctx,
                    product,
                    [_footprint_representation(ctx, element.params.outline, element.params.base_z)],
                )
            _pset(ctx, product, "VegetationArea", element, {"BaseZ": element.params.base_z, "TriangleCount": len(surface[1]) if surface else 0})
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
    apply_outdoor_styles(ctx)


_OUTDOOR_ELEMENT_TYPES = (
    BuildingElement, CurbElement, FenceElement, ManholeElement, MarkingElement, PavementElement, ShrubElement,
    SiteWallElement, TerrainElement, TreeElement, VegetationAreaElement,
)
_PAVEMENT_CATEGORY = {"Strasse": "road", "Trottoir": "sidewalk", "Weg": "path", "Randstein": "curb"}
_GEOGRAPHIC_CATEGORY = {"Baum": "tree", "Busch": "shrub", "Vegetationsfläche": "lawn"}


def outdoor_category_of_product(product: Entity) -> str | None:
    """Outdoor style category of an IFC product (None when the product is not an outdoor element)."""
    kind = str(getattr(product, "PredefinedType", None) or "").upper()
    object_type = str(getattr(product, "ObjectType", None) or "")
    if product.is_a("IfcGeographicElement"):
        return "terrain" if kind == "TERRAIN" else _GEOGRAPHIC_CATEGORY.get(object_type)
    if product.is_a("IfcCivilElement"):
        return _PAVEMENT_CATEGORY.get(object_type)
    if product.is_a("IfcRailing"):
        return "fence"
    if product.is_a("IfcDistributionChamberElement"):
        return "manhole"
    if product.is_a("IfcAnnotation"):
        return "marking"
    if product.is_a("IfcWall"):
        return "site_wall" if object_type == "Mauer" else "building_wall"
    if product.is_a("IfcRoof") or (product.is_a("IfcSlab") and kind == "ROOF"):
        return "roof"
    if product.is_a("IfcWindow"):
        return "window"
    if product.is_a("IfcDoor"):
        return "door"
    if product.is_a("IfcOpeningElement"):
        for relation in getattr(product, "HasFillings", None) or ():
            filling = relation.RelatedBuildingElement
            if filling.is_a("IfcDoor"):
                return "door"
        return "window"
    return None


def apply_outdoor_styles(ctx: ExportContext) -> int:
    """Colour and layer every outdoor product by category (``config/outdoor_style.yaml``).

    Each representation item gets an IfcStyledItem (IfcSurfaceStyle with rendering, IfcCurveStyle for
    curves) and each shape representation joins exactly one IfcPresentationLayerAssignment named like
    the DXF layer of its category. Scenes without outdoor elements (indoor export) are left untouched.
    Returns the number of styled products.
    """
    if not any(isinstance(element, _OUTDOOR_ELEMENT_TYPES) for element in ctx.scene.elements.values()):
        return 0
    f = ctx.file
    styles = load_outdoor_style()
    surface: dict[str, Entity] = {}
    curve: dict[str, Entity] = {}
    layers: dict[str, Entity] = {}
    count = 0
    for product in f.by_type("IfcProduct"):
        representation = getattr(product, "Representation", None)
        if representation is None:
            continue
        category = outdoor_category_of_product(product)
        style = styles.get(category) if category else None
        if style is None or not style["ifc"]:
            continue
        if category not in surface:
            surface[category] = shapes.surface_style(f, style["label"], style["rgb"], style["transparency"])
            curve[category] = shapes.curve_style(f, style["label"], style["rgb"])
        layer = shapes.presentation_layer(f, layers, style["layer"])
        for shape in representation.Representations:
            for item in shape.Items:
                if item.is_a("IfcGeometricCurveSet"):
                    for element in item.Elements:
                        shapes.style_item(f, element, curve[category])
                elif item.is_a("IfcCurve"):
                    shapes.style_item(f, item, curve[category])
                else:
                    shapes.style_item(f, item, surface[category])
            shapes.add_to_layer(layer, shape)
        count += 1
    return count
