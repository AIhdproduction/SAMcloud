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
from samcloud.drone.contracts import BuildingModel, LocalFrame, Opening, QAEntry

LAYER_BUILDING = "S-BLDG"
LAYER_ROOF = "S-BLDG-ROOF"
LAYER_WINDOW = "S-BLDG-WIND"
LAYER_DOOR = "S-BLDG-DOOR"


def _status(value: str) -> Status:
    return {"ok": Status.OK, "lod1_fallback": Status.WARNING}.get(value, Status.FAIL)


def _facade_axes(model: BuildingModel, facades) -> list[tuple[np.ndarray, np.ndarray, float]]:
    """Return stable facade segments, favouring supplied fitted facade surfaces."""
    result: list[tuple[np.ndarray, np.ndarray, float]] = []
    for facade in facades or ():
        segment = getattr(facade, "segment", None)
        if segment is not None and len(segment.coords) >= 2:
            a, b = np.asarray(segment.coords[0], float), np.asarray(segment.coords[-1], float)
            if np.linalg.norm(b - a) > 1e-6:
                result.append((a, b, float(getattr(facade, "z_min", np.min(model.vertices[:, 2])))))
    if result:
        return result
    vertices, faces, roles = np.asarray(model.vertices, float), np.asarray(model.faces, int), np.asarray(model.face_roles)
    edges: dict[tuple[int, int], int] = {}
    for face, role in zip(faces, roles, strict=True):
        if role != 1:
            continue
        for a, b in ((face[0], face[1]), (face[1], face[2]), (face[2], face[0])):
            key = tuple(sorted((int(a), int(b))))
            edges[key] = edges.get(key, 0) + 1
    for (i, j), count in edges.items():
        a, b = vertices[i], vertices[j]
        if count == 1 and abs(a[2] - b[2]) < 0.15 and np.linalg.norm(a[:2] - b[:2]) > 0.1:
            result.append((a[:2], b[:2], float(min(a[2], b[2]))))
    return result


def _wall_elements(model: BuildingModel, facades) -> tuple[list[WallElement], list[tuple[np.ndarray, np.ndarray, float]]]:
    axes = _facade_axes(model, facades)
    height = max(float(np.ptp(model.vertices[:, 2])), 0.1)
    walls: list[WallElement] = []
    for index, (a, b, base) in enumerate(axes):
        walls.append(WallElement(
            id=f"building-{model.building_id}-facade-{index}", name=f"Building {model.building_id} facade {index}",
            building_id=f"building-{model.building_id}", params=WallParams(
                start=tuple(a), end=tuple(b), thickness=0.20, height=height, base_offset=base,
                justification=WallJustification.CENTER, function=WallFunction.EXTERIOR,
            ), param_origin={"start": ParamOrigin.FITTED, "end": ParamOrigin.FITTED, "thickness": ParamOrigin.ASSUMED,
                           "height": ParamOrigin.FITTED, "base_offset": ParamOrigin.FITTED}, status=_status(model.status),
        ))
    return walls, axes


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
        walls, axes = _wall_elements(model, (facades_by_building or {}).get(bid, ()))
        elements.extend(walls)
        for ordinal, opening in enumerate((openings_by_building or {}).get(bid, ())):
            oid = f"building-{bid}-opening-{ordinal}"
            if opening.status not in ("ok", "warning") or opening.u1 <= opening.u0 or opening.z1 <= opening.z0:
                qa.append(QAEntry("opening", oid, opening.reason or "rejected opening"))
                continue
            if opening.facade_index < 0 or opening.facade_index >= len(walls):
                qa.append(QAEntry("opening", oid, "facade index has no host wall"))
                continue
            width, height = float(opening.u1 - opening.u0), float(opening.z1 - opening.z0)
            host = walls[opening.facade_index]
            opening_el = OpeningElement(
                id=oid, name=f"{opening.kind} opening", building_id=host.building_id,
                params=OpeningParams(host_wall_id=host.id, position_along_axis=max(0.0, float(opening.u0)), width=width,
                                     height=height, sill_height=max(0.0, float(opening.z0 - axes[opening.facade_index][2])),
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
    return elements, qa


def _ring_from_mesh(element: BuildingElement) -> np.ndarray:
    v, f, r = np.asarray(element.params.vertices, float), np.asarray(element.params.triangles, int), np.asarray(element.params.face_roles)
    ids = np.unique(f[r == 0].ravel()) if np.any(r == 0) else np.arange(len(v))
    pts = v[ids]
    c = pts[:, :2].mean(axis=0)
    order = np.argsort(np.arctan2(pts[:, 1] - c[1], pts[:, 0] - c[0]))
    return pts[order]


def to_dxf_entities(elements: Sequence[object], frame: LocalFrame) -> dict[str, list[dict]]:
    """Return LV95 CAD entities for building envelopes, roof edges and opening plan symbols."""
    out: dict[str, list[dict]] = {}
    building_axes: dict[str, tuple[np.ndarray, np.ndarray, float]] = {}
    for el in elements:
        if isinstance(el, BuildingElement):
            ring = frame.to_lv95(_ring_from_mesh(el))
            out.setdefault(LAYER_BUILDING, []).append({"type": "lwpolyline", "points": ring.tolist(), "closed": True, "elevation": float(np.min(ring[:, 2]))})
            v, f, r = np.asarray(el.params.vertices, float), np.asarray(el.params.triangles, int), np.asarray(el.params.face_roles)
            roof_edges: set[tuple[int, int]] = set()
            for face in f[r == 2]:
                for a, b in ((face[0], face[1]), (face[1], face[2]), (face[2], face[0])): roof_edges.add(tuple(sorted((int(a), int(b)))))
            for a, b in roof_edges:
                out.setdefault(LAYER_ROOF, []).append({"type": "polyline3d", "points": frame.to_lv95(v[[a, b]]).tolist(), "closed": False})
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
