"""Building envelopes, facade openings and their outdoor CAD representation."""
from __future__ import annotations

from collections.abc import Sequence
import math
import numpy as np

from samcloud.bim.core.model import (
    BuildingElement, BuildingParams, DoorElement, DoorParams, OpeningElement, OpeningParams,
    ParamOrigin, Status, WallElement, WallFunction, WallJustification, WallParams,
    WindowElement, WindowParams,
)
from samcloud.buildings.facades import facade_surfaces
from samcloud.drone.contracts import BuildingModel, LocalFrame, Opening, QAEntry

LAYER_BUILDING = "S-BLDG"
LAYER_ROOF = "S-BLDG-ROOF"
LAYER_WINDOW = "S-BLDG-WIND"
LAYER_DOOR = "S-BLDG-DOOR"


def _status(value: str) -> Status:
    return {"ok": Status.OK, "lod1_fallback": Status.WARNING}.get(value, Status.FAIL)


MIN_FACADE_LENGTH = 0.5  # shorter wall pieces (stair-step risers) are merged into their neighbour


def _facade_surface_list(model: BuildingModel, facades) -> list:
    """Return mesh-exact facade surfaces (one per coplanar LoD2 wall face group)."""
    given = [f for f in (facades or ()) if hasattr(f, "direction_xy") and hasattr(f, "face_indices")]
    if given:
        return given
    try:
        return facade_surfaces(model, [])
    except Exception:  # noqa: BLE001 - a bad wall mesh must not drop the building
        return []


def _merge_small_surfaces(model: BuildingModel, surfaces: list, protected: set[int]) -> list[tuple[object, list[int], int]]:
    """Group surfaces into walls: ``(primary surface, face indices, member surface positions)``.

    Pieces shorter than ``MIN_FACADE_LENGTH`` that host no opening join the longest touching surface.
    """
    faces = np.asarray(model.faces, int)
    verts = [set(int(v) for i in s.face_indices for v in faces[i]) for s in surfaces]
    target = list(range(len(surfaces)))
    for pos, surface in enumerate(surfaces):
        if pos in protected or float(surface.length) >= MIN_FACADE_LENGTH:
            continue
        best = None
        for other, osurf in enumerate(surfaces):
            if other == pos or float(osurf.length) < MIN_FACADE_LENGTH or not (verts[pos] & verts[other]):
                continue
            if best is None or osurf.length > surfaces[best].length:
                best = other
        if best is not None:
            target[pos] = best
    groups: dict[int, list[int]] = {}
    for pos, primary in enumerate(target):
        groups.setdefault(primary, []).append(pos)
    result = []
    for primary, members in groups.items():
        indices = sorted({int(i) for m in members for i in surfaces[m].face_indices})
        result.append((surfaces[primary], indices, members))
    return result


def _wall_elements(model: BuildingModel, facades, protected: set[int] | None = None) -> tuple[list[WallElement], dict[int, int]]:
    """One WallElement per facade plane; returns the walls and a surface-position -> wall-index map."""
    surfaces = _facade_surface_list(model, facades)
    vertices = np.asarray(model.vertices, float)
    faces = np.asarray(model.faces, int)
    walls: list[WallElement] = []
    index_map: dict[int, int] = {}
    for primary, indices, members in _merge_small_surfaces(model, surfaces, protected or set()):
        pts = vertices[faces[indices].ravel()]
        z_min, z_max = float(pts[:, 2].min()), float(pts[:, 2].max())
        direction = np.asarray(primary.direction_xy, float)
        origin = np.asarray(primary.origin_xy, float)
        length = max(float(primary.length), float(((pts[:, :2] - origin) @ direction).max()))
        if length <= 1e-6:
            continue
        index = len(walls)
        for member in members:
            index_map[member] = index
        walls.append(WallElement(
            id=f"building-{model.building_id}-facade-{index}", name=f"Building {model.building_id} facade {index}",
            building_id=f"building-{model.building_id}", params=WallParams(
                start=tuple(float(v) for v in origin), end=tuple(float(v) for v in origin + direction * length),
                thickness=0.20, height=max(z_max - z_min, 0.1), base_offset=z_min,
                justification=WallJustification.CENTER, function=WallFunction.EXTERIOR,
            ), param_origin={"start": ParamOrigin.FITTED, "end": ParamOrigin.FITTED, "thickness": ParamOrigin.ASSUMED,
                           "height": ParamOrigin.FITTED, "base_offset": ParamOrigin.FITTED},
            evidence={"face_indices": indices}, status=_status(model.status),
        ))
    return walls, index_map


def _merged_wall(model: BuildingModel, facade_walls: list[WallElement]) -> WallElement:
    """The single exterior wall of a building: every facade face in one element.

    Its axis is that of the longest facade (only a reference line); the IFC body is the facade mesh.
    """
    faces = np.asarray(model.faces, int)
    vertices = np.asarray(model.vertices, float)
    indices = [int(i) for i in np.flatnonzero(np.asarray(model.face_roles) == 1)]
    points = vertices[faces[indices].ravel()] if indices else vertices
    z_min, z_max = float(points[:, 2].min()), float(points[:, 2].max())
    reference = max(facade_walls, key=lambda w: float(np.hypot(w.params.end[0] - w.params.start[0], w.params.end[1] - w.params.start[1])), default=None)
    if reference is not None:
        start, end = reference.params.start, reference.params.end
    else:
        low = points[:, :2]
        start, end = tuple(float(v) for v in low.min(axis=0)), tuple(float(v) for v in low.max(axis=0))
        if np.hypot(end[0] - start[0], end[1] - start[1]) < 1e-3:
            end = (start[0] + 1.0, start[1])
    bid = int(model.building_id)
    return WallElement(
        id=f"building-{bid}-wall", name=f"Fassade {bid}", building_id=f"building-{bid}", params=WallParams(
            start=tuple(float(v) for v in start), end=tuple(float(v) for v in end),
            thickness=0.20, height=max(z_max - z_min, 0.1), base_offset=z_min,
            justification=WallJustification.CENTER, function=WallFunction.EXTERIOR,
        ), param_origin={"start": ParamOrigin.FITTED, "end": ParamOrigin.FITTED, "thickness": ParamOrigin.ASSUMED,
                       "height": ParamOrigin.FITTED, "base_offset": ParamOrigin.FITTED},
        evidence={"face_indices": indices, "facade_count": len(facade_walls)}, status=_status(model.status),
    )


def build_building_elements(
    models: list[BuildingModel], openings_by_building: dict[int, list[Opening]], facades_by_building,
    *, config: dict | None = None,
) -> tuple[list[object], list[QAEntry]]:
    """Build LoD envelopes plus IFC-hosted accepted windows and doors.

    Facade ``WallElement`` records provide a proper IFC host for void/fill relations while the
    ``BuildingElement`` remains the authoritative measured LoD envelope.
    """
    elements: list[object] = []
    qa: list[QAEntry] = []
    for model in models:
        bid = int(model.building_id)
        if model.status == "failed":
            qa.append(QAEntry("building", f"building-{bid}", model.reason or "reconstruction failed"))
            continue
        vertices, faces = np.asarray(model.vertices, float), np.asarray(model.faces, int)
        if len(vertices) < 4 or len(faces) == 0:
            qa.append(QAEntry("building", f"building-{bid}", "empty building mesh"))
            continue
        evidence = {"building_id": bid, "lod": model.lod, "coverage": model.coverage, "rmse": model.rmse}
        building = BuildingElement(
            id=f"building-{bid}", tag=str(bid), name=f"Building {bid}",
            params=BuildingParams(vertices=[tuple(map(float, p)) for p in vertices], triangles=[tuple(map(int, f)) for f in faces],
                                 face_roles=[int(v) for v in np.asarray(model.face_roles)]),
            param_origin={"vertices": ParamOrigin.FITTED, "triangles": ParamOrigin.FITTED, "face_roles": ParamOrigin.FITTED},
            source_instance_ids=[str(bid)], evidence={k: v for k, v in evidence.items() if v is not None}, status=_status(model.status),
            notes=[model.reason] if model.reason else [],
        )
        building.metrics.deviation_rms = model.rmse
        building.metrics.coverage = model.coverage
        elements.append(building)
        hosting = {int(o.facade_index) for o in (openings_by_building or {}).get(bid, ()) if o.facade_index >= 0}
        walls, index_map = _wall_elements(model, (facades_by_building or {}).get(bid, ()), hosting)
        merged = _merged_wall(model, walls)
        used_hosts: set[int] = set()
        for ordinal, opening in enumerate((openings_by_building or {}).get(bid, ())):
            oid = f"building-{bid}-opening-{ordinal}"
            if opening.status not in ("ok", "warning") or opening.u1 <= opening.u0 or opening.z1 <= opening.z0:
                qa.append(QAEntry("opening", oid, opening.reason or "rejected opening"))
                continue
            if opening.facade_index not in index_map:
                qa.append(QAEntry("opening", oid, "facade index has no host wall"))
                continue
            width, height = float(opening.u1 - opening.u0), float(opening.z1 - opening.z0)
            used_hosts.add(index_map[opening.facade_index])
            host = walls[index_map[opening.facade_index]]
            opening_el = OpeningElement(
                id=oid, name=f"{opening.kind} opening", building_id=host.building_id,
                params=OpeningParams(host_wall_id=host.id, position_along_axis=max(0.0, float(opening.u0)), width=width,
                                     height=height, sill_height=max(0.0, float(opening.z0 - host.params.base_offset)),
                                     reveal_depth=host.params.thickness),
                param_origin={"host_wall_id": ParamOrigin.FITTED, "position_along_axis": ParamOrigin.FITTED,
                              "width": ParamOrigin.FITTED, "height": ParamOrigin.FITTED, "sill_height": ParamOrigin.FITTED,
                              "reveal_depth": ParamOrigin.ASSUMED}, evidence={"views": opening.evidence.views, "points": opening.evidence.points},
                status=_status(opening.status),
            )
            elements.append(opening_el)
            common = dict(id=f"{oid}-{opening.kind}", name=opening.kind.title(), building_id=host.building_id,
                          param_origin={"opening_id": ParamOrigin.FITTED}, status=_status(opening.status))
            if opening.kind == "door":
                elements.append(DoorElement(params=DoorParams(opening_id=oid), **common))
            elif opening.kind == "window":
                elements.append(WindowElement(params=WindowParams(opening_id=oid), **common))
            else:
                qa.append(QAEntry("opening", oid, f"unknown opening kind {opening.kind}"))
        elements.append(merged)
        for position in sorted(used_hosts):  # facade walls survive only as opening hosts
            host = walls[position]
            host.evidence = {**{k: v for k, v in host.evidence.items() if k != "face_indices"}, "host_only": True, "merged_wall": merged.id}
            host.name = f"{host.name} (opening host)"
            elements.append(host)
    return elements, qa


def _ring_from_mesh(element: BuildingElement) -> np.ndarray:
    v, f, r = np.asarray(element.params.vertices, float), np.asarray(element.params.triangles, int), np.asarray(element.params.face_roles)
    ids = np.unique(f[r == 0].ravel()) if np.any(r == 0) else np.arange(len(v))
    pts = v[ids]
    c = pts[:, :2].mean(axis=0)
    order = np.argsort(np.arctan2(pts[:, 1] - c[1], pts[:, 0] - c[0]))
    return pts[order]


DXF_SIMPLIFY_M = 0.01  # Douglas-Peucker tolerance of merged roof lines


def _simplify(points: np.ndarray, tolerance: float) -> np.ndarray:
    """Douglas-Peucker on a 3D polyline (open chain or closed loop whose last point repeats the first)."""
    if len(points) <= 2:
        return points
    start, end = points[0], points[-1]
    chord = end - start
    norm = float(np.linalg.norm(chord))
    if norm < 1e-9:  # closed loop: split at the farthest point
        index = int(np.argmax(np.linalg.norm(points - start, axis=1)))
        if index == 0:
            return points[:1]
        return np.vstack([_simplify(points[: index + 1], tolerance)[:-1], _simplify(points[index:], tolerance)])
    distance = np.linalg.norm(np.cross(points[1:-1] - start, chord / norm), axis=1)
    if len(distance) == 0 or float(distance.max()) <= tolerance:
        return np.vstack([start, end])
    index = int(np.argmax(distance)) + 1
    return np.vstack([_simplify(points[: index + 1], tolerance)[:-1], _simplify(points[index:], tolerance)])


def roof_lines(vertices: np.ndarray, roof_faces: np.ndarray, *, coplanar_deg: float = 2.0, tolerance: float = DXF_SIMPLIFY_M) -> list[np.ndarray]:
    """Real roof lines of a triangulated roof: outline, ridges, hips, valleys and steps.

    An edge is kept when it borders only one roof face (outline) or when its two roof faces differ in
    normal by more than ``coplanar_deg``; coplanar triangle seams give nothing.  Edges are chained through
    vertices of degree two and simplified (Douglas-Peucker, ``tolerance``).
    """
    vertices = np.asarray(vertices, float)
    faces = np.asarray(roof_faces, int)
    if len(faces) == 0:
        return []
    normals = np.cross(vertices[faces[:, 1]] - vertices[faces[:, 0]], vertices[faces[:, 2]] - vertices[faces[:, 0]])
    lengths = np.linalg.norm(normals, axis=1)
    normals = normals / np.where(lengths > 1e-12, lengths, 1.0)[:, None]
    owners: dict[tuple[int, int], list[int]] = {}
    for index, face in enumerate(faces):
        for a, b in ((face[0], face[1]), (face[1], face[2]), (face[2], face[0])):
            if a != b:
                owners.setdefault((min(int(a), int(b)), max(int(a), int(b))), []).append(index)
    cosine = float(np.cos(np.deg2rad(coplanar_deg)))
    kept: set[tuple[int, int]] = set()
    for edge, faces_at in owners.items():
        if len(faces_at) != 2 or lengths[faces_at[0]] < 1e-12 or lengths[faces_at[1]] < 1e-12 or float(normals[faces_at[0]] @ normals[faces_at[1]]) < cosine:
            kept.add(edge)
    adjacency: dict[int, set[int]] = {}
    for a, b in kept:
        adjacency.setdefault(a, set()).add(b)
        adjacency.setdefault(b, set()).add(a)
    used: set[tuple[int, int]] = set()

    def walk(first: int, second: int) -> list[int]:
        chain = [first, second]
        used.add((min(first, second), max(first, second)))
        while len(adjacency[chain[-1]]) == 2 and chain[-1] != chain[0]:
            nxt = next(n for n in adjacency[chain[-1]] if n != chain[-2])
            key = (min(chain[-1], nxt), max(chain[-1], nxt))
            if key in used:
                break
            used.add(key)
            chain.append(nxt)
        return chain

    chains: list[list[int]] = []
    for node in sorted(adjacency):
        if len(adjacency[node]) != 2:
            for other in sorted(adjacency[node]):
                if (min(node, other), max(node, other)) not in used:
                    chains.append(walk(node, other))
    for node in sorted(adjacency):  # pure loops (every vertex has degree two)
        for other in sorted(adjacency[node]):
            if (min(node, other), max(node, other)) not in used:
                chains.append(walk(node, other))
    return [_simplify(vertices[chain], tolerance) for chain in chains if len(chain) >= 2]


def to_dxf_entities(elements: Sequence[object], frame: LocalFrame, *, coplanar_deg: float | None = None) -> dict[str, list[dict]]:
    """Return LV95 CAD entities for building envelopes, roof edges and opening plan symbols."""
    if coplanar_deg is None:
        from samcloud.drone.contracts import load_drone_config
        coplanar_deg = float(load_drone_config()["buildings"].get("dxf", {}).get("coplanar_deg", 2.0))
    out: dict[str, list[dict]] = {}
    building_axes: dict[str, tuple[np.ndarray, np.ndarray, float]] = {}
    for el in elements:  # opening hosts first: their order in the element list is irrelevant
        if isinstance(el, WallElement) and el.id.startswith("building-"):
            building_axes[el.id] = (np.asarray(el.params.start, float), np.asarray(el.params.end, float), float(el.params.thickness))
    for el in elements:
        if isinstance(el, BuildingElement):
            ring = frame.to_lv95(_ring_from_mesh(el))
            out.setdefault(LAYER_BUILDING, []).append({"type": "lwpolyline", "points": ring.tolist(), "closed": True, "elevation": float(np.min(ring[:, 2]))})
            v, f, r = np.asarray(el.params.vertices, float), np.asarray(el.params.triangles, int), np.asarray(el.params.face_roles)
            for chain in roof_lines(v, f[r == 2], coplanar_deg=coplanar_deg):
                out.setdefault(LAYER_ROOF, []).append({"type": "polyline3d", "points": frame.to_lv95(chain).tolist(), "closed": False})
        elif isinstance(el, WallElement) and el.id.startswith("building-"):
            building_axes[el.id] = (np.asarray(el.params.start, float), np.asarray(el.params.end, float), float(el.params.thickness))
        elif isinstance(el, OpeningElement):
            axis = building_axes.get(el.params.host_wall_id)
            if axis is None:
                continue
            a, b, thickness = axis; direction = b - a; length = float(np.linalg.norm(direction))
            if length <= 1e-9: continue
            direction /= length; normal = np.array([-direction[1], direction[0]])
            p0, p1 = a + direction * el.params.position_along_axis, a + direction * (el.params.position_along_axis + el.params.width)
            front = np.vstack([p0 + normal * thickness / 2, p1 + normal * thickness / 2]); back = np.vstack([p0 - normal * thickness / 2, p1 - normal * thickness / 2])
            points = frame.to_lv95(np.column_stack([front, np.full(2, el.params.sill_height)])).tolist() + frame.to_lv95(np.column_stack([back, np.full(2, el.params.sill_height)])).tolist()
            filling = next((x for x in elements if getattr(getattr(x, "params", None), "opening_id", None) == el.id), None)
            if isinstance(filling, WindowElement):
                out.setdefault(LAYER_WINDOW, []).append({"type": "line", "start": points[0], "end": points[1]})
                out.setdefault(LAYER_WINDOW, []).append({"type": "line", "start": points[2], "end": points[3]})
            elif isinstance(filling, DoorElement):
                centre = frame.to_lv95(np.array([p0[0], p0[1], el.params.sill_height])).tolist()
                end = frame.to_lv95(np.array([p1[0], p1[1], el.params.sill_height])).tolist()
                out.setdefault(LAYER_DOOR, []).append({"type": "line", "start": centre, "end": end})
                out.setdefault(LAYER_DOOR, []).append({"type": "arc", "center": centre, "radius": el.params.width, "start_angle": math.degrees(math.atan2(direction[1], direction[0])) - 90, "end_angle": math.degrees(math.atan2(direction[1], direction[0]))})
    return out


def category_counts(elements: Sequence[object]) -> dict[str, int]:
    return {"building": sum(isinstance(x, BuildingElement) for x in elements), "window": sum(isinstance(x, WindowElement) for x in elements), "door": sum(isinstance(x, DoorElement) for x in elements)}
