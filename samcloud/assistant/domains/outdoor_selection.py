"""Element lists, details, 2D picks, outlines and IFC ids of the outdoor assistant domain.

Positions for the model are LV95 (E/N, metres) and LN02 heights; :func:`hit_test`, :func:`outline` and the selection
dicts work in LOCAL metres of the Scene (the viewers convert with ``view_frame()["origin_offset"]``).

Picks: manhole cover circle, tree crown circle, fence / wall / curb polyline within :data:`LINE_PICK_M` (walls: half the
thickness more), building and pavement footprints (read-only in tier ``"elements"`` but selectable for questions);
anything else is a point.  IFC GlobalIds are the deterministic ``guid_from_uuid(uuid_of(element.id))`` of the outdoor
IFC writer (the gutter stone of a curb is ``<curb id>-gutter``).
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from samcloud.assistant.domains.outdoor_edits import (
    building_heights, element_anchor, element_height, find_element, is_manual, kind_of, mesh_footprint, polyline_length,
)
from samcloud.assistant.domains.outdoor_scene import origin_offset

LINE_PICK_M = 0.3
POINT_PICK_MIN_M = 0.3
LIST_KINDS = ("tree", "manhole", "fence", "site_wall", "curb", "pavement", "building", "terrain", "wall", "shrub", "marking",
              "vegetation_area")
PICK_ORDER = ("manhole", "tree", "fence", "site_wall", "curb", "building", "pavement")
GUTTER_SUFFIX = "-gutter"

_CHIP_FALLBACK = {
    "de": {"tree": "Baum {id} ({shape}, {height} m)", "manhole": "Schacht {id} (Ø {diameter} m)", "fence": "Zaun {id} ({height} m)",
           "site_wall": "Mauer {id} ({height} m)", "curb": "Randstein {id} ({length} m)", "pavement": "Belag {id} ({pavement_kind})",
           "building": "Gebäude {id}", "terrain": "Gelände", "wall": "Fassade {id}", "element": "{id}",
           "point": "Punkt E {e} / N {n}"},
    "en": {"tree": "Tree {id} ({shape}, {height} m)", "manhole": "Manhole {id} (Ø {diameter} m)", "fence": "Fence {id} ({height} m)",
           "site_wall": "Wall {id} ({height} m)", "curb": "Curb {id} ({length} m)", "pavement": "Pavement {id} ({pavement_kind})",
           "building": "Building {id}", "terrain": "Terrain", "wall": "Facade {id}", "element": "{id}",
           "point": "Point E {e} / N {n}"},
}
_SHAPES = {"de": {"sphere": "Kugel", "ellipsoid": "Ellipsoid", "cone": "Kegel", "umbrella": "Schirm", "cylinder": "Säule"},
           "en": {"sphere": "sphere", "ellipsoid": "ellipsoid", "cone": "cone", "umbrella": "umbrella", "cylinder": "column"}}
_PAVEMENT = {"de": {"road": "Strasse", "sidewalk": "Trottoir", "path": "Weg"}, "en": {"road": "road", "sidewalk": "sidewalk", "path": "path"}}


def _r(value: float, digits: int = 3) -> float:
    return float(round(float(value), digits))


def _en(scene: Any, x: float, y: float) -> list[float]:
    offset = origin_offset(scene)
    return [_r(float(x) + offset[0]), _r(float(y) + offset[1])]


def _ln02(scene: Any, z: float) -> float:
    return _r(float(z) + float(origin_offset(scene)[2]))


# ---------------------------------------------------------------------------------------------------- element lists

def element_row(scene: Any, element: Any, ref: str | None = None) -> dict[str, Any]:
    """Compact row of one element (LV95 position, metres); ``ref`` = its short display label (``"B9"``)."""
    kind, p = kind_of(element), element.params
    row: dict[str, Any] = {"id": element.id, "type": kind}
    if ref is None:
        from samcloud.assistant.domains.outdoor_labels import display_label

        ref = display_label(scene, element.id)
    if ref:
        row["ref"] = ref
    if element.name:
        row["label"] = element.name
    anchor = element_anchor(element)
    if anchor is not None:
        row["position"] = _en(scene, *anchor)
    height = element_height(element)
    if height is not None:
        row["height"] = _r(height, 2)
    if kind == "tree":
        row.update({"crown_diameter": _r(p.crown_diameter, 2), "trunk_diameter": _r(p.trunk_diameter, 2),
                    "crown_shape": p.crown_shape or "ellipsoid"})
    elif kind == "manhole":
        row.update({"diameter": _r(p.diameter, 2), "cover_ln02": _ln02(scene, p.center[2])})
    elif kind in ("fence", "site_wall", "curb"):
        row["length"] = _r(polyline_length(p.axis), 2) if p.axis is not None else None
        if kind == "curb":
            row.update({"width": _r(p.width, 2), "gutter_width": _r(p.gutter_width, 2)})
        elif kind == "fence":
            row["thickness"] = _r(p.thickness, 2)
        elif p.thickness is not None:
            row["thickness"] = _r(p.thickness, 2)
    elif kind in ("pavement", "building", "terrain"):
        footprint = mesh_footprint(element) if getattr(p, "vertices", None) else None
        if footprint is not None:
            row["area_m2"] = _r(footprint.area, 1)
        if kind == "pavement":
            row["pavement_kind"] = p.kind
    if is_manual(element):
        row["evidence"] = "manual"
    return row


def element_list(scene: Any, spec: Any = None) -> list[dict[str, Any]]:
    """Rows of the elements; ``spec`` = a category name or ``{"category" / "kind", "label"}``."""
    if isinstance(spec, str):
        spec = {"category": spec}
    spec = dict(spec or {})
    category = str(spec.get("category") or spec.get("kind") or "").strip().lower()
    label = str(spec.get("label") or "").strip().lower()
    from samcloud.assistant.domains.outdoor_labels import display_labels

    refs = display_labels(scene)
    rows = []
    for element in scene.elements.values():
        kind = kind_of(element)
        if kind not in LIST_KINDS or (category and kind != category):
            continue
        ref = refs.get(element.id, "")
        if label and label not in element.id.lower() and label not in str(element.name or "").lower() and label != ref.lower():
            continue
        rows.append(element_row(scene, element, ref))
    return rows


def element_details(scene: Any, element_id: str) -> dict[str, Any]:
    """Everything about one element (LV95 / LN02, metres); ``{"found": False}`` when unknown."""
    element = find_element(scene, element_id)
    if element is None:
        return {"found": False, "id": str(element_id)}
    from samcloud.assistant.domains.outdoor_edits import snapshot

    kind, p = kind_of(element), element.params
    details: dict[str, Any] = {"found": True, **element_row(scene, element), **snapshot(scene, element)}
    details.pop("kind", None)
    details["param_origin"] = {name: getattr(origin, "value", origin) for name, origin in element.param_origin.items()}
    details["provenance"] = getattr(element.provenance, "value", element.provenance)
    details["status"] = getattr(element.status, "value", element.status)
    if element.notes:
        details["notes"] = list(element.notes)[:6]
    evidence = {k: v for k, v in element.evidence.items() if isinstance(v, (str, int, float, bool))}
    if evidence:
        details["evidence"] = evidence
    if kind in ("fence", "site_wall", "curb") and p.axis is not None:
        details["axis"] = [_en(scene, q[0], q[1]) for q in p.axis]
        details["direction_note"] = "offset distance_m > 0 moves it to the left looking from the first to the last vertex"
        if kind == "fence":
            details["top_ln02"] = [_ln02(scene, q[2]) for q in p.axis]
            if p.bottom_z is not None:
                details["bottom_ln02"] = [_ln02(scene, z) for z in p.bottom_z]
        elif kind == "curb":
            details["top_ln02"] = [_ln02(scene, q[2]) for q in p.axis]
            if p.foot_z is not None:
                details["foot_ln02"] = [_ln02(scene, z) for z in p.foot_z]
    if kind == "building":
        heights = building_heights(element)
        details.update({"eave_height": _r(heights["eave"] - heights["ground"], 2), "ridge_height": _r(heights["ridge"] - heights["ground"], 2)})
        details["facade_elements"] = [e.id for e in scene.elements.values() if getattr(e, "building_id", None) == element.id and e.id != element.id][:20]
    return details


# -------------------------------------------------------------------------------------------------------- selection

def selection_for(scene: Any, element_id: str) -> dict[str, Any]:
    """Selection dict of an element (``{"type": "unknown", "id"}`` when it does not exist)."""
    element = find_element(scene, element_id)
    if element is None:
        return {"type": "unknown", "id": str(element_id)}
    kind, p = kind_of(element), element.params
    selection: dict[str, Any] = {"type": "element", "kind": kind, "id": element.id, "label": element.name or element.id}
    from samcloud.assistant.domains.outdoor_labels import display_label

    ref = display_label(scene, element.id)
    if ref:
        selection["ref"] = ref
    height = element_height(element)
    if height is not None:
        selection["height_m"] = _r(height, 2)
    if kind == "tree":
        selection["shape"] = p.crown_shape or "ellipsoid"
        selection["crown_diameter_m"] = _r(p.crown_diameter, 2)
    elif kind == "manhole":
        selection["diameter_m"] = _r(p.diameter, 2)
    elif kind in ("fence", "site_wall", "curb") and p.axis is not None:
        selection["length_m"] = _r(polyline_length(p.axis), 2)
    elif kind == "pavement":
        selection["pavement_kind"] = p.kind
    return selection


def _segment_distance(points: np.ndarray, x: float, y: float) -> float:
    a, b = points[:-1], points[1:]
    ab = b - a
    t = np.clip(((x - a[:, 0]) * ab[:, 0] + (y - a[:, 1]) * ab[:, 1]) / np.maximum((ab ** 2).sum(axis=1), 1e-12), 0.0, 1.0)
    closest = a + ab * t[:, None]
    return float(np.hypot(closest[:, 0] - x, closest[:, 1] - y).min())


def hit_test(scene: Any, x: float, y: float) -> dict[str, Any]:
    """Element under a click at local (x, y), else ``{"type": "point", "x", "y"}``."""
    best: dict[str, tuple[float, str]] = {}
    for element in scene.elements.values():
        kind, p = kind_of(element), element.params
        score = None
        if kind == "manhole":
            radius = max(float(p.diameter) / 2.0, POINT_PICK_MIN_M)
            distance = math.hypot(float(p.center[0]) - x, float(p.center[1]) - y)
            score = distance / radius if distance <= radius else None
        elif kind == "tree":
            offset = p.crown_offset or (0.0, 0.0)
            cx, cy = float(p.trunk_xy[0]) + float(offset[0]), float(p.trunk_xy[1]) + float(offset[1])
            radius = max(float(p.crown_diameter) / 2.0, POINT_PICK_MIN_M)
            distance = math.hypot(cx - x, cy - y)
            score = distance / radius if distance <= radius else None
        elif kind in ("fence", "site_wall", "curb") and getattr(p, "axis", None):
            reach = LINE_PICK_M + (float(p.thickness or 0.0) / 2.0 if kind == "site_wall" else 0.0)
            pts = np.asarray([(float(q[0]), float(q[1])) for q in p.axis], dtype=np.float64)
            distance = _segment_distance(pts, x, y) if len(pts) > 1 else math.inf
            score = distance / reach if distance <= reach else None
        elif kind == "site_wall" and getattr(p, "footprint", None):
            from shapely.geometry import Point, Polygon

            score = 0.5 if Polygon(p.footprint).buffer(LINE_PICK_M).contains(Point(x, y)) else None
        elif kind in ("building", "pavement"):
            from shapely.geometry import Point

            footprint = mesh_footprint(element)
            score = 0.5 if footprint is not None and footprint.contains(Point(x, y)) else None
        if score is not None and (kind not in best or score < best[kind][0]):
            best[kind] = (score, element.id)
    for kind in PICK_ORDER:
        if kind in best:
            return selection_for(scene, best[kind][1])
    return {"type": "point", "x": _r(x), "y": _r(y)}


def chip_label(sel: dict[str, Any], lang: str | None = None, scene: Any = None) -> str:
    """Short chip text (``assistant.outdoor.chip.*`` when present, built-in de/en otherwise)."""
    language = (lang or "de").lower()[:2]
    language = language if language in _CHIP_FALLBACK else "en"
    if sel.get("type") == "point":
        x, y = float(sel.get("x", 0.0)), float(sel.get("y", 0.0))
        if scene is not None:
            e, n = _en(scene, x, y)
        else:
            e, n = x, y
        key, values = "point", {"e": f"{e:.2f}", "n": f"{n:.2f}"}
    else:
        key = str(sel.get("kind") or "element")
        shape = str(sel.get("shape") or "ellipsoid")
        values = {"id": sel.get("ref") or sel.get("id", ""), "label": sel.get("label", ""),
                  "height": "" if sel.get("height_m") is None else f"{float(sel['height_m']):.2f}",
                  "diameter": "" if sel.get("diameter_m") is None else f"{float(sel['diameter_m']):.2f}",
                  "length": "" if sel.get("length_m") is None else f"{float(sel['length_m']):.1f}",
                  "shape": _SHAPES[language].get(shape, shape),
                  "pavement_kind": _PAVEMENT[language].get(str(sel.get("pavement_kind") or ""), str(sel.get("pavement_kind") or ""))}
    template = None
    try:
        from samcloud.app.i18n import catalogue

        template = catalogue(language).get(f"assistant.outdoor.chip.{key}")
    except Exception:  # noqa: BLE001 - the built-in texts are enough without the UI catalogues
        template = None
    if not template:
        template = _CHIP_FALLBACK[language].get(key, _CHIP_FALLBACK[language]["element"])
    try:
        return str(template).format(**values)
    except (KeyError, IndexError, ValueError):
        return str(sel.get("label") or sel.get("id") or "")


def outline(scene: Any, sel: dict[str, Any]) -> list[list[list[float]]]:
    """Closed 2D rings (local metres) of a selected element for the overlays."""
    element = find_element(scene, sel.get("id")) if isinstance(sel, dict) else None
    if element is None:
        return []
    return element_rings(element)


def element_rings(element: Any) -> list[list[list[float]]]:
    kind, p = kind_of(element), element.params

    def circle(cx: float, cy: float, r: float, n: int = 24) -> list[list[float]]:
        return [[_r(cx + r * math.cos(2 * math.pi * i / n)), _r(cy + r * math.sin(2 * math.pi * i / n))] for i in range(n)]

    if kind == "tree":
        offset = p.crown_offset or (0.0, 0.0)
        return [circle(float(p.trunk_xy[0]) + float(offset[0]), float(p.trunk_xy[1]) + float(offset[1]), float(p.crown_diameter) / 2.0)]
    if kind == "manhole":
        return [circle(float(p.center[0]), float(p.center[1]), max(float(p.diameter) / 2.0, 0.1), 16)]
    if kind == "site_wall" and getattr(p, "footprint", None):
        return [[[_r(x), _r(y)] for x, y in p.footprint]]
    if kind in ("fence", "site_wall", "curb") and getattr(p, "axis", None):
        from shapely.geometry import LineString

        width = {"fence": max(float(getattr(p, "thickness", 0.04) or 0.04), 0.1), "curb": max(float(getattr(p, "width", 0.12)), 0.1)}.get(kind, 0.2)
        band = LineString([(float(q[0]), float(q[1])) for q in p.axis]).buffer(width / 2.0, cap_style=2, join_style=2)
        polygons = [band] if band.geom_type == "Polygon" else list(getattr(band, "geoms", []))
        return [[[_r(x), _r(y)] for x, y in list(poly.exterior.coords)[:-1]] for poly in polygons if not poly.is_empty]
    if kind in ("building", "pavement") and getattr(p, "vertices", None):
        footprint = mesh_footprint(element)
        if footprint is None:
            return []
        polygons = [footprint] if footprint.geom_type == "Polygon" else [g for g in getattr(footprint, "geoms", []) if g.geom_type == "Polygon"]
        return [[[_r(x), _r(y)] for x, y in list(poly.exterior.simplify(0.02).coords)[:-1]] for poly in polygons]
    return []


def z_range(scene: Any, sel: dict[str, Any]) -> tuple[float, float] | None:
    """Absolute height range (LN02, the IFC frame) of a selected element."""
    element = find_element(scene, sel.get("id")) if isinstance(sel, dict) else None
    if element is None:
        return None
    return element_z_range(scene, element)


def element_z_range(scene: Any, element: Any) -> tuple[float, float] | None:
    kind, p = kind_of(element), element.params
    dz = float(origin_offset(scene)[2])
    if kind == "tree":
        return _r(float(p.base_z) + dz), _r(float(p.base_z) + float(p.height) + dz)
    if kind == "manhole":
        return _r(float(p.center[2]) - float(p.thickness) + dz), _r(float(p.center[2]) + dz)
    if kind == "fence":
        top = np.asarray(p.axis, dtype=np.float64)[:, 2]
        bottom = np.asarray(p.bottom_z, dtype=np.float64) if p.bottom_z is not None else top - float(p.height)
        return _r(float(bottom.min()) + dz), _r(float(top.max()) + dz)
    if kind == "site_wall":
        return _r(float(p.base_z) + dz), _r(float(p.top_z) + dz)
    if kind == "curb":
        top = np.asarray(p.axis, dtype=np.float64)[:, 2]
        foot = np.asarray(p.foot_z, dtype=np.float64) if p.foot_z is not None else top - float(p.height)
        return _r(float(foot.min()) - float(p.embed) + dz), _r(float(top.max()) + dz)
    if getattr(p, "vertices", None):
        z = np.asarray(p.vertices, dtype=np.float64)[:, 2]
        return _r(float(z.min()) + dz), _r(float(z.max()) + dz)
    return None


def global_id_of(element_id: str) -> str:
    from samcloud.bim.core.ifc.ids import guid_from_uuid, uuid_of

    return guid_from_uuid(uuid_of(str(element_id)))


def global_id_map(scene: Any) -> dict[str, str]:
    """{IFC GlobalId: element id} of every element (a curb's gutter stone maps to its curb)."""
    mapping: dict[str, str] = {}
    for element in scene.elements.values():
        mapping[global_id_of(element.id)] = element.id
        if kind_of(element) == "curb":
            mapping[global_id_of(element.id + GUTTER_SUFFIX)] = element.id
    return mapping


__all__ = ["chip_label", "element_details", "element_list", "element_rings", "element_row", "element_z_range", "global_id_map",
           "global_id_of", "hit_test", "outline", "selection_for", "z_range"]
