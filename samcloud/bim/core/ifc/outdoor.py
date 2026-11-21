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
    """Footprint of the whole axis polyline: mitre joins, flat caps, counter-clockwise."""
    xy = [(float(p[0]), float(p[1])) for p in axis]
    polygon = shapely.geometry.LineString(xy).buffer(thickness / 2.0, cap_style=2, join_style=2)
    if polygon.geom_type != "Polygon":
        polygon = max((g for g in getattr(polygon, "geoms", []) if g.geom_type == "Polygon"), key=lambda g: g.area)
    ring = shapely.geometry.polygon.orient(Polygon(polygon.exterior), 1.0).exterior.coords
    return [(float(x), float(y)) for x, y in list(ring)[:-1]]


def _mitre_vectors(xy: np.ndarray, limit: float = 3.0) -> np.ndarray:
    """Per-vertex left offset vectors (unit distance, mitre joins) of a plan polyline."""
    count = len(xy)
    seg = np.diff(xy, axis=0)
    length = np.hypot(seg[:, 0], seg[:, 1])
    normals = np.zeros_like(seg)
    good = length > 1e-9
    normals[good] = np.column_stack([-seg[good, 1], seg[good, 0]]) / length[good, None]
    if not good.all():  # zero-length segments take the nearest valid normal
        valid = np.flatnonzero(good)
        if len(valid) == 0:
            normals[:] = (0.0, 1.0)
        else:
            for i in np.flatnonzero(~good):
                normals[i] = normals[valid[np.argmin(np.abs(valid - i))]]
    vectors = np.zeros((count, 2))
    vectors[0], vectors[-1] = normals[0], normals[-1]
    for i in range(1, count - 1):
        n1, n2 = normals[i - 1], normals[i]
        vec = (n1 + n2) / max(1.0 + float(n1 @ n2), 1e-6)
        size = float(np.hypot(*vec))
        vectors[i] = vec if size <= limit else vec / size * limit
    return vectors


def _ribbon_mesh(
    xy: np.ndarray, d_left: float, d_right: float, z_top: np.ndarray, z_bottom: np.ndarray,
    z_top_right: np.ndarray | None = None, z_bottom_right: np.ndarray | None = None,
) -> tuple[list[Vec3], list[tuple[int, int, int]]]:
    """Closed, upright solid along a plan polyline.

    Its sides lie ``d_left`` / ``d_right`` (signed, left normal positive) beside the polyline; per station the
    top is ``z_top`` and the bottom ``z_bottom``. Every vertex of the top stays above the bottom of its own station.
    """
    xy = np.asarray(xy, dtype=float)[:, :2]
    n = len(xy)
    vec = _mitre_vectors(xy)
    left, right = xy + vec * d_left, xy + vec * d_right
    top, bottom = np.asarray(z_top, dtype=float), np.asarray(z_bottom, dtype=float)
    top_r = top if z_top_right is None else np.asarray(z_top_right, dtype=float)  # a draped top differs per side
    bottom_r = bottom if z_bottom_right is None else np.asarray(z_bottom_right, dtype=float)
    vertices: list[Vec3] = (
        [(float(left[i, 0]), float(left[i, 1]), float(top[i])) for i in range(n)]
        + [(float(right[i, 0]), float(right[i, 1]), float(top_r[i])) for i in range(n)]
        + [(float(left[i, 0]), float(left[i, 1]), float(bottom[i])) for i in range(n)]
        + [(float(right[i, 0]), float(right[i, 1]), float(bottom_r[i])) for i in range(n)]
    )
    lt, rt, lb, rb = (lambda i: i), (lambda i: n + i), (lambda i: 2 * n + i), (lambda i: 3 * n + i)
    triangles: list[tuple[int, int, int]] = []
    for i in range(n - 1):
        j = i + 1
        triangles += [
            (rt(i), rt(j), lt(j)), (rt(i), lt(j), lt(i)),      # top
            (rb(i), lb(j), rb(j)), (rb(i), lb(i), lb(j)),      # bottom
            (lb(i), lt(j), lb(j)), (lb(i), lt(i), lt(j)),      # left side
            (rb(i), rb(j), rt(j)), (rb(i), rt(j), rt(i)),      # right side
        ]
    last = n - 1
    triangles += [(rb(0), lt(0), lb(0)), (rb(0), rt(0), lt(0)), (rb(last), lb(last), lt(last)), (rb(last), lt(last), rt(last))]
    return vertices, triangles


def _wall_mesh(
    ring: Sequence[tuple[float, float]], base: float, top: float, axis: Sequence[tuple[float, float]] | None,
    top_axis: Sequence[float] | None, base_axis: Sequence[float] | None = None,
) -> tuple[list[Vec3], list[tuple[int, int, int]]]:
    """Closed solid of a wall footprint whose top (and base) height follows the axis (edges per station)."""
    polygon = shapely.geometry.polygon.orient(Polygon(ring), 1.0)
    pts = np.asarray(polygon.exterior.coords, dtype=float)[:-1]
    count = len(pts)
    z_top = np.full(count, float(top))
    if axis is not None and top_axis is not None and len(axis) == len(top_axis) and len(axis) >= 2:
        line = shapely.geometry.LineString([(float(p[0]), float(p[1])) for p in axis])
        cum = np.r_[0.0, np.cumsum(np.hypot(*np.diff(np.asarray(axis, dtype=float)[:, :2], axis=0).T))]
        z_top = np.interp(shapely.line_locate_point(line, shapely.points(pts)), cum, np.asarray(top_axis, dtype=float))
    z_base = np.full(count, float(base))
    if axis is not None and base_axis is not None and len(axis) == len(base_axis) and len(axis) >= 2:
        line_b = shapely.geometry.LineString([(float(p[0]), float(p[1])) for p in axis])
        cum_b = np.r_[0.0, np.cumsum(np.hypot(*np.diff(np.asarray(axis, dtype=float)[:, :2], axis=0).T))]
        z_base = np.interp(shapely.line_locate_point(line_b, shapely.points(pts)), cum_b, np.asarray(base_axis, dtype=float))
    z_top = np.maximum(z_top, z_base + 1e-3)
    index = {(round(float(x), 6), round(float(y), 6)): i for i, (x, y) in enumerate(pts)}
    vertices: list[Vec3] = [(float(x), float(y), float(z)) for (x, y), z in zip(pts, z_top)] + [(float(x), float(y), float(z)) for (x, y), z in zip(pts, z_base)]
    triangles: list[tuple[int, int, int]] = []
    for tri in shapely.constrained_delaunay_triangles(polygon).geoms if hasattr(shapely, "constrained_delaunay_triangles") else []:
        ids = [index.get((round(float(x), 6), round(float(y), 6))) for x, y in list(tri.exterior.coords)[:3]]
        if None in ids:
            continue
        a, b, c = ids
        area = (pts[b, 0] - pts[a, 0]) * (pts[c, 1] - pts[a, 1]) - (pts[c, 0] - pts[a, 0]) * (pts[b, 1] - pts[a, 1])
        if area < 0:
            b, c = c, b
        triangles += [(a, b, c), (count + a, count + c, count + b)]
    for i in range(count):
        j = (i + 1) % count
        triangles += [(count + i, count + j, j), (count + i, j, i)]
    return vertices, triangles


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


def _curb_body_heights(curb: CurbElement) -> np.ndarray:
    """Body height per axis vertex: from the curb top down to ``embed`` below the road-side foot level."""
    axis = np.asarray(curb.params.axis, dtype=float)
    foot = curb.params.foot_z
    if foot is None or len(foot) != len(axis):
        return np.full(len(axis), float(curb.params.height))
    return np.maximum(axis[:, 2] - (np.asarray(foot, dtype=float) - curb.params.embed), 0.05)


_MITRE_Z_LIMIT_M = 0.004  # the mitre extension of a curb segment may rise at most this far above its axis


def _curb_extrusions(ctx: ExportContext, curb: CurbElement) -> list[Entity]:
    """Straight curb segments as extrusions whose top edge is the axis and whose body hangs below it."""
    axis = [np.asarray(point, dtype=float) for point in curb.params.axis]
    heights = _curb_body_heights(curb)
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
                slope = abs(float(vector[2])) / max(math.hypot(float(vector[0]), float(vector[1])), 1e-9)
                if slope > 1e-9:  # the mitre overshoot climbs along the segment: keep it within the QA height tolerance
                    extension = min(extension, _MITRE_Z_LIMIT_M / slope)
                if target == "start":
                    extension_start = extension
                else:
                    extension_end = extension
        origin = start - direction * extension_start
        height = float((heights[index] + heights[index + 1]) / 2.0)
        profile = shapes.rectangle_profile(ctx.file, (-height / 2.0, 0.0), height, curb.params.width)
        solids.append(shapes.extrusion(ctx.file, profile, length + extension_start + extension_end, origin=tuple(origin), axis=tuple(direction), ref_direction=(0.0, 0.0, 1.0)))
    return solids


def _gutter_mesh(curb: CurbElement) -> tuple[list[Vec3], list[tuple[int, int, int]]] | None:
    """Gutter stone (Wasserstein): flat slab on the road side, top at the foot level, none without gutter data."""
    p = curb.params
    dense = p.gutter_axis is not None and len(p.gutter_axis) >= 2
    if not p.gutter_width or not p.road_side or (not dense and (p.foot_z is None or len(p.foot_z) != len(p.axis))):
        return None
    xy = np.asarray(p.gutter_axis if dense else p.axis, dtype=float)[:, :2]
    foot = np.asarray(p.gutter_axis, dtype=float)[:, 2] if dense else np.asarray(p.foot_z, dtype=float)
    inner, outer = p.road_side * p.width / 2.0, p.road_side * (p.width / 2.0 + p.gutter_width)
    if dense and p.gutter_z_outer is not None and len(p.gutter_z_outer) == len(foot):
        z_outer = np.asarray(p.gutter_z_outer, dtype=float)  # draped: curb-side and outer edge each at the surface height there
        z_left, z_right = (z_outer, foot) if outer > inner else (foot, z_outer)
        return _ribbon_mesh(xy, max(inner, outer), min(inner, outer), z_left, z_left - p.gutter_thickness, z_right, z_right - p.gutter_thickness)
    return _ribbon_mesh(xy, max(inner, outer), min(inner, outer), foot, foot - p.gutter_thickness)


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
    """One IfcWall per building whose body holds every LoD2 facade face.

    The generic wall writer extrudes a flat prism; here its body is replaced by the mesh faces listed in
    the merged wall's ``face_indices`` evidence, so the wall ends exactly at the LoD2 roof edge and the
    terrain-following bottom.  Facade walls that only host openings (``host_only``) are dissolved: their
    openings void the merged wall.  Returns the number of walls written.
    """
    count = 0
    triangles = list(building.params.triangles)
    merged = None
    hosts: list[WallElement] = []
    for element in ctx.scene.elements.values():
        if not isinstance(element, WallElement) or element.building_id != building.id:
            continue
        if element.evidence.get("host_only"):
            hosts.append(element)
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
        merged = wall
    if merged is not None:
        for element in hosts:
            host = ctx.entities.pop(element.id, None)
            if host is None:
                continue
            for relation in list(getattr(host, "HasOpenings", None) or ()):
                relation.RelatingBuildingElement = merged
            ifcopenshell.api.root.remove_product(ctx.file, product=host)
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
                ifcopenshell.api.spatial.assign_container(ctx.file, products=[roof], relating_structure=building)
                shapes.place(ctx, roof, (0.0, 0.0, 0.0))
                _assign_mesh(ctx, roof, element.params.vertices, roofs)  # one IfcRoof, one body with every roof face
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
            tp = element.params
            trunk_measured = tp.trunk_measured if tp.trunk_measured is not None else getattr(element.param_origin.get("trunk_diameter"), "value", "measured") != "assumed"
            trunk, crown = shapes.stylised_tree(ctx.file, tp.trunk_xy, tp.base_z, tp.height, tp.crown_diameter, tp.trunk_diameter,
                                                tp.crown_shape or "ellipsoid", tp.crown_base_z, tp.crown_offset or (0.0, 0.0))
            shapes.assign_representations(ctx, product, [shapes.tessellation_representation(ctx, [trunk, crown])])  # ONE Body: viewers use only the first
            _pset(ctx, product, "Tree", element, {
                "Height": tp.height, "CrownDiameter": tp.crown_diameter, "CrownShape": tp.crown_shape or "ellipsoid",
                "TrunkDiameter": tp.trunk_diameter if trunk_measured else "nicht gemessen",
                "TrunkDiameterMeasured": bool(trunk_measured)})
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
            if element.params.top_z_axis is not None or element.params.base_z_axis is not None:  # top or base varies along the wall: triangulated solid
                _assign_mesh(ctx, product, *_wall_mesh(outline, element.params.base_z, element.params.top_z, element.params.axis, element.params.top_z_axis, element.params.base_z_axis))
            else:
                profile = shapes.polygon_profile(ctx.file, outline)
                solid = shapes.extrusion(ctx.file, profile, element.params.top_z - element.params.base_z, origin=(0.0, 0.0, element.params.base_z))
                shapes.assign_representations(ctx, product, [shapes.swept_solid_representation(ctx, [solid])])
            _pset(ctx, product, "SiteWall", element, {"BaseZ": element.params.base_z, "TopZ": element.params.top_z, "Thickness": element.params.thickness or 0.0})
        elif isinstance(element, FenceElement):
            product = _product(ctx, element, "IfcRailing", "Fence")
            product.ObjectType = "Zaun"
            fence_axis = np.asarray(element.params.axis, dtype=float)  # top edge; the body stays upright between top and bottom edge
            bottom = np.asarray(element.params.bottom_z, dtype=float) if element.params.bottom_z is not None and len(element.params.bottom_z) == len(fence_axis) else fence_axis[:, 2] - element.params.height
            half = element.params.thickness / 2.0
            _assign_mesh(ctx, product, *_ribbon_mesh(fence_axis[:, :2], half, -half, np.maximum(fence_axis[:, 2], bottom), bottom))
            _pset(ctx, product, "Fence", element, {"Height": element.params.height, "Thickness": element.params.thickness})
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
                solid = shapes.swept_curb(ctx.file, element.params.axis, element.params.width, float(np.median(_curb_body_heights(element))), down=True)
                shapes.assign_representations(ctx, product, [shapes.swept_solid_representation(ctx, [solid])])
            else:
                shapes.assign_representations(ctx, product, [shapes.swept_solid_representation(ctx, _curb_extrusions(ctx, element))])
            _pset(ctx, product, "Curb", element, {"Width": element.params.width, "Height": float(np.median(_curb_body_heights(element))), "Sweep": element.params.curb_sweep, "RoadSide": element.params.road_side})
            gutter = _gutter_mesh(element)
            if gutter is not None:  # gutter stone (Wasserstein) as its own product beside the curb
                gutter_element = element.model_copy(update={"id": f"{element.id}-gutter", "name": f"Wasserstein {element.tag or element.id[:8]}"})
                gutter_product = _product(ctx, gutter_element, "IfcCivilElement", "Gutter")
                gutter_product.ObjectType = "Wasserstein"
                _assign_mesh(ctx, gutter_product, *gutter)
                _pset(ctx, gutter_product, "Gutter", gutter_element, {"Width": element.params.gutter_width, "Thickness": element.params.gutter_thickness})
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
            cover_z = element.params.center[2]  # body top = cover height, so terrain never covers it
            solid = shapes.extrusion(ctx.file, profile, element.params.thickness, origin=(element.params.center[0], element.params.center[1], cover_z - element.params.thickness))
            shapes.assign_representations(ctx, product, [shapes.swept_solid_representation(ctx, [solid])])
            _pset(ctx, product, "Manhole", element, {"Diameter": element.params.diameter, "Z_DECKEL": element.params.center[2], "Type": element.params.type})
    apply_outdoor_styles(ctx)


_OUTDOOR_ELEMENT_TYPES = (
    BuildingElement, CurbElement, FenceElement, ManholeElement, MarkingElement, PavementElement, ShrubElement,
    SiteWallElement, TerrainElement, TreeElement, VegetationAreaElement,
)
_PAVEMENT_CATEGORY = {"Strasse": "road", "Trottoir": "sidewalk", "Weg": "path", "Randstein": "curb", "Wasserstein": "gutter"}
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
