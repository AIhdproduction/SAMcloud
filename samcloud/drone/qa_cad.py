"""Read-only CAD/BIM rule checker for finished outdoor DXF and IFC files.

Every check returns a list of result dicts ``{rule, ok, value, threshold, offenders}``
(offenders capped at 20). Nothing here ever modifies the inspected files.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import ezdxf
import numpy as np
from shapely.geometry import Polygon
from shapely.ops import unary_union

from samcloud.core.paths import CONFIG_DIRECTORY
from samcloud.drone.contracts import load_drone_config

logger = logging.getLogger(__name__)

MAX_OFFENDERS = 20
IDS_PATH = Path(CONFIG_DIRECTORY) / "bim" / "samcloud.ids"

BLOCK_RULES: dict[str, tuple[str, tuple[str, ...]]] = {
    "S-TREE": ("TREE", ("HEIGHT", "CROWN_DIA", "TRUNK_DIA")),
    "S-SHRUB": ("SHRUB", ("HEIGHT",)),
    "S-MANH": ("MANHOLE", ("Z_DECKEL", "TYP")),
}
# Additional blocks that may sit on a block-rule layer (not counted as the main object): trunk circle of a tree.
EXTRA_BLOCKS: dict[str, tuple[str, ...]] = {"S-TREE": ("TREE_STEM",)}
# Owner rule "Busch: Block oder geschlossene Umrisslinie mit HEIGHT": closed outlines are also allowed on these layers.
OUTLINE_ALLOWED_LAYERS = ("S-SHRUB",)
AREA_LAYERS = ("S-PAVE", "S-WALK", "S-PATH", "S-VEG")
TIN_LAYER = "S-TOPO-TIN"
CONTOUR_LAYERS = ("S-TOPO-MAJR", "S-TOPO-MINR")
MAJOR_LAYER = "S-TOPO-MAJR"
CONTOUR_LABEL_LAYER = "S-TOPO-TEXT"
CONTOUR_LABEL_REACH_M = 0.5   # a label belongs to a contour when its anchor lies this close to the line
CURB_LAYER = "S-CURB"
MESH_TYPES = ("3DFACE", "MESH")
CURB_MIN_SEGMENT_M = 0.05
LV95_E = (2_480_000.0, 2_840_000.0)
LV95_N = (1_070_000.0, 1_300_000.0)
MAX_PLACEMENT_DISTANCE_M = 10_000.0

# Keyword tables (lower case, matched against IFC Name / ObjectType prefix) and DXF layers per category.
# Real writer strings (samcloud/bim/core/ifc/outdoor.py): ObjectType Baum / Busch / Vegetationsfläche / Mauer / Zaun /
# Strasse / Trottoir / Weg / Randstein / Markierung; Names "Tree ...", "Shrub ...", "Vegetation area ...", "Site wall ...",
# "Fence ...", "Pavement ...", "Curb ...", "Marking ...", "Manhole ...". Older spellings stay as aliases.
IFC_CATEGORY_KEYWORDS: dict[str, tuple[str, ...]] = {
    "tree": ("baum", "tree"),
    "shrub": ("busch", "shrub", "strauch"),
    "vegetation_area": ("vegetationsfläche", "vegetationsflaeche", "vegetation area", "vegetationsflaeche"),
    "manhole": ("schacht", "kanaldeckel", "manhole"),
    "curb": ("randstein", "curb"),
    "road": ("strasse", "road"),
    "sidewalk": ("trottoir", "sidewalk"),
    "path": ("weg", "path"),
    "site_wall": ("mauer", "site wall"),
    "fence": ("zaun", "fence"),
    "marking": ("markierung", "marking"),
    "gutter": ("wasserstein", "gutter"),
}
# IFC classes searched per category (default: IFC_DEFAULT_CLASSES).
IFC_CATEGORY_CLASSES: dict[str, tuple[str, ...]] = {
    "tree": ("IfcGeographicElement",),
    "shrub": ("IfcGeographicElement",),
    "vegetation_area": ("IfcGeographicElement",),
    "site_wall": ("IfcWall",),
    "fence": ("IfcRailing",),
    "marking": ("IfcAnnotation",),
    "manhole": ("IfcDistributionChamberElement",),
    "building": ("IfcBuilding",),
    "window": ("IfcWindow",),
    "door": ("IfcDoor",),
}
IFC_DEFAULT_CLASSES = ("IfcCivilElement", "IfcBuiltElement", "IfcBuildingElementProxy", "IfcDistributionChamberElement", "IfcGeographicElement")
DXF_CATEGORY_LAYERS: dict[str, str] = {
    "tree": "S-TREE", "shrub": "S-SHRUB", "manhole": "S-MANH", "curb": "S-CURB",
    "road": "S-PAVE", "sidewalk": "S-WALK", "path": "S-PATH",
    "building": "S-BLDG", "window": "S-BLDG-WIND", "door": "S-BLDG-DOOR", "site_wall": "S-WALL",
    "fence": "S-FENC", "marking": "S-MARK", "vegetation_area": "S-VEG", "gutter": "S-GUTR",
}
PAVEMENT_KINDS = ("road", "sidewalk", "path")


def _result(rule: str, ok: bool, value: Any, threshold: Any, offenders: Iterable[Any] = ()) -> dict[str, Any]:
    return {
        "rule": rule, "ok": bool(ok), "value": value, "threshold": threshold,
        "offenders": [str(o) for o in list(offenders)[:MAX_OFFENDERS]],
    }


def _cfg(config: Mapping[str, Any] | None) -> dict[str, Any]:
    return dict(config) if config is not None else load_drone_config()


# ============================================================================== DXF


def _is_closed(entity) -> bool:
    if entity.dxftype() == "LWPOLYLINE":
        return bool(entity.closed)
    if entity.dxftype() == "POLYLINE":
        return bool(entity.is_closed)
    return False


def _is_polyface(entity) -> bool:
    return entity.dxftype() == "POLYLINE" and bool(entity.is_poly_face_mesh)


def _label(entity) -> str:
    return f"{entity.dxftype()}@{entity.dxf.layer}#{entity.dxf.handle}"


def _entity_points(entity) -> list[tuple[float, float]]:
    kind = entity.dxftype()
    try:
        if kind == "LWPOLYLINE":
            return [(p[0], p[1]) for p in entity.get_points("xy")]
        if kind == "POLYLINE":
            return [(v.dxf.location.x, v.dxf.location.y) for v in entity.vertices]
        if kind in ("INSERT", "CIRCLE", "POINT", "TEXT"):
            attr = "center" if kind == "CIRCLE" else ("location" if kind == "POINT" else "insert")
            loc = getattr(entity.dxf, attr)
            return [(loc.x, loc.y)]
        if kind == "3DFACE":
            return [(v.x, v.y) for v in entity.wcs_vertices()]
        if kind == "SPLINE":
            pts = list(entity.fit_points) or list(entity.control_points)
            return [(p[0], p[1]) for p in pts]
        if kind == "LINE":
            return [(entity.dxf.start.x, entity.dxf.start.y), (entity.dxf.end.x, entity.dxf.end.y)]
    except Exception:  # noqa: BLE001 - exotic entity, skip silently
        return []
    return []


def _polyline_xy(entity) -> np.ndarray:
    return np.asarray(_entity_points(entity), dtype=float).reshape(-1, 2)


def _rule_block_inserts(msp) -> dict[str, Any]:
    offenders: list[str] = []
    counts: dict[str, int] = {}
    for layer, (block, attrs) in BLOCK_RULES.items():
        counts[layer] = 0
        for entity in msp.query(f'*[layer=="{layer}"]'):
            counts[layer] += 1
            if layer in OUTLINE_ALLOWED_LAYERS and entity.dxftype() in ("LWPOLYLINE", "POLYLINE") and _is_closed(entity):
                continue
            if entity.dxftype() == "INSERT" and entity.dxf.name in EXTRA_BLOCKS.get(layer, ()):
                continue
            if entity.dxftype() != "INSERT" or entity.dxf.name != block:
                offenders.append(f"{_label(entity)} not INSERT {block}")
                continue
            tags = {a.dxf.tag.upper() for a in entity.attribs}
            missing = [a for a in attrs if a not in tags]
            if missing:
                offenders.append(f"{_label(entity)} missing attributes {','.join(missing)}")
        for entity in msp.query("INSERT"):
            if entity.dxf.name == block and entity.dxf.layer != layer:
                offenders.append(f"{_label(entity)} block {block} on wrong layer")
    return _result("1_block_inserts", not offenders, counts, "INSERT TREE/SHRUB/MANHOLE with attributes", offenders)


def _rule_area_layers(msp) -> dict[str, Any]:
    offenders: list[str] = []
    n = 0
    for layer in AREA_LAYERS:
        for entity in msp.query(f'*[layer=="{layer}"]'):
            n += 1
            kind = entity.dxftype()
            if kind == "HATCH":
                continue
            if kind in ("LWPOLYLINE", "POLYLINE") and not _is_polyface(entity) and _is_closed(entity):
                continue
            offenders.append(_label(entity))
    return _result("2_area_layers_closed_polylines", not offenders, n, "closed (LW)POLYLINE/HATCH only", offenders)


def _rule_tin(doc, msp, cfg: Mapping[str, Any]) -> dict[str, Any]:
    offenders = [_label(e) for e in msp if (e.dxftype() in MESH_TYPES or _is_polyface(e)) and e.dxf.layer != TIN_LAYER]
    problems = list(offenders)
    state = "absent"
    if TIN_LAYER in doc.layers:
        layer = doc.layers.get(TIN_LAYER)
        frozen = bool(layer.is_frozen())
        off = bool(layer.is_off())
        state = "frozen" if frozen else ("off" if off else "visible")
        if cfg.get("dxf", {}).get("tin_layer_frozen", True) and not (frozen or off):
            problems.append(f"layer {TIN_LAYER} neither frozen nor off")
    return _result("3_tin_only_on_tin_layer_frozen", not problems, {"foreign_meshes": len(offenders), "tin_layer": state},
                   f"meshes only on {TIN_LAYER}; layer frozen/off", problems)


def _rule_contours(msp, cfg: Mapping[str, Any]) -> dict[str, Any]:
    want_spline = bool(cfg.get("dxf", {}).get("spline_contours", True))
    offenders: list[str] = []
    n = 0
    for layer in CONTOUR_LAYERS:
        for entity in msp.query(f'*[layer=="{layer}"]'):
            n += 1
            kind = entity.dxftype()
            ok = kind == "SPLINE" if want_spline else kind in ("SPLINE", "LWPOLYLINE", "POLYLINE")
            if not ok:
                offenders.append(_label(entity))
    return _result("4_contours_spline", not offenders, n, "SPLINE" if want_spline else "SPLINE/POLYLINE", offenders)


def _contour_polyline(entity) -> np.ndarray | None:
    kind = entity.dxftype()
    if kind == "SPLINE":
        points = [(float(p[0]), float(p[1])) for p in entity.flattening(0.1)]
    elif kind == "LWPOLYLINE":
        points = [(float(p[0]), float(p[1])) for p in entity.get_points("xy")]
    elif kind == "POLYLINE":
        points = [(float(v.dxf.location[0]), float(v.dxf.location[1])) for v in entity.vertices]
    else:
        return None
    return np.asarray(points, dtype=float) if len(points) >= 2 else None


def _label_anchor(entity) -> tuple[float, float] | None:
    if entity.dxftype() not in ("TEXT", "MTEXT"):
        return None
    point = entity.dxf.align_point if entity.dxftype() == "TEXT" and entity.dxf.hasattr("align_point") else entity.dxf.insert
    return float(point[0]), float(point[1])


def _rule_contour_labels(msp, cfg: Mapping[str, Any]) -> dict[str, Any]:
    """Every major contour longer than ``dxf.contour_label_spacing_m`` carries at least one height label (S-TOPO-TEXT)."""
    spacing = float(cfg.get("dxf", {}).get("contour_label_spacing_m", 10.0))
    lines: list[tuple[str, Any]] = []
    for entity in msp.query(f'*[layer=="{MAJOR_LAYER}"]'):
        xy = _contour_polyline(entity)
        if xy is not None and len(xy) >= 2:
            from shapely.geometry import LineString

            line = LineString(xy)
            if line.length > spacing:
                lines.append((_label(entity), line))
    anchors = [a for a in (_label_anchor(e) for e in msp.query(f'*[layer=="{CONTOUR_LABEL_LAYER}"]')) if a is not None]
    offenders: list[str] = []
    if lines:
        import shapely

        points = shapely.points(np.asarray(anchors, dtype=float).reshape(-1, 2)) if anchors else np.empty(0, dtype=object)
        tree = shapely.STRtree(points) if len(points) else None
        for name, line in lines:
            if tree is None or not len(tree.query(line, predicate="dwithin", distance=CONTOUR_LABEL_REACH_M)):
                offenders.append(name)
    return _result("15_contours_labelled", not offenders, {"major_lines_over_spacing": len(lines), "labels": len(anchors)},
                   f"every major contour longer than {spacing:g} m has a label", offenders)


def _curb_polylines(msp) -> list[tuple[str, np.ndarray]]:
    out = []
    for entity in msp.query(f'*[layer=="{CURB_LAYER}"]'):
        if entity.dxftype() in ("LWPOLYLINE", "POLYLINE") and not _is_polyface(entity):
            out.append((_label(entity), _polyline_xy(entity)))
    return out


def _collinear_triples(xy: np.ndarray, tol: float) -> list[tuple[int, float, float]]:
    """Return (middle index, length a-b, length b-c) of every near-collinear consecutive triple."""
    found = []
    for i in range(1, len(xy) - 1):
        a, b, c = xy[i - 1], xy[i], xy[i + 1]
        chord = c - a
        length = float(np.hypot(*chord))
        dist = float(np.hypot(*(b - a))) if length < 1e-12 else abs(chord[0] * (b - a)[1] - chord[1] * (b - a)[0]) / length
        if dist <= tol:
            found.append((i, float(np.hypot(*(b - a))), float(np.hypot(*(c - b)))))
    return found


def _rules_curb(msp, cfg: Mapping[str, Any]) -> list[dict[str, Any]]:
    tol = float(cfg.get("curb", {}).get("dp_tolerance_m", 0.015))
    triples = 0
    collinear = 0
    off_col: list[str] = []
    off_dense: list[str] = []
    for label, xy in _curb_polylines(msp):
        found = _collinear_triples(xy, tol)
        triples += max(len(xy) - 2, 0)
        collinear += len(found)
        if found:
            off_col.append(f"{label}: {len(found)} collinear vertices")
        short = [f for f in found if min(f[1], f[2]) < CURB_MIN_SEGMENT_M]
        if short:
            off_dense.append(f"{label}: {len(short)} segments < {CURB_MIN_SEGMENT_M} m")
    share = collinear / triples if triples else 0.0
    return [
        _result("5_curb_collinear_share", collinear == 0, round(share, 4), 0.0, off_col),
        _result("5_curb_vertex_density", not off_dense, len(off_dense), f"<1 vertex per {CURB_MIN_SEGMENT_M} m", off_dense),
    ]


def _rule_coordinates(doc, msp, cfg: Mapping[str, Any]) -> dict[str, Any]:
    local = bool(cfg.get("dxf", {}).get("local_origin", False))
    if local:
        base = doc.header.get("$INSBASE", (0.0, 0.0, 0.0))
        ok = any(abs(float(v)) > 0.0 for v in base)
        return _result("6_coordinates_plausible", ok, [float(v) for v in base], "$INSBASE set (local origin)",
                       [] if ok else ["$INSBASE is zero"])
    offenders: list[str] = []
    n = 0
    for entity in msp:
        for x, y in _entity_points(entity):
            n += 1
            if not (LV95_E[0] <= x <= LV95_E[1] and LV95_N[0] <= y <= LV95_N[1]):
                offenders.append(f"{_label(entity)} at ({x:.2f}, {y:.2f})")
                break
    return _result("6_coordinates_plausible", not offenders, n, f"LV95 E{LV95_E} N{LV95_N}", offenders)


def check_dxf(path, config: Mapping[str, Any] | None = None) -> list[dict[str, Any]]:
    """Check a DXF file against rules 1-6 (blocks, area layers, TIN, contours, curbs, coordinates)."""
    cfg = _cfg(config)
    doc = ezdxf.readfile(str(path))
    msp = doc.modelspace()
    results = [_rule_block_inserts(msp), _rule_area_layers(msp), _rule_tin(doc, msp, cfg), _rule_contours(msp, cfg)]
    results.extend(_rules_curb(msp, cfg))
    results.append(_rule_coordinates(doc, msp, cfg))
    results.append(_rule_contour_labels(msp, cfg))
    return results


# ============================================================================== IFC


def _text_matches(element, keywords: Sequence[str]) -> bool:
    for text in (getattr(element, "Name", None), getattr(element, "ObjectType", None)):
        if not text:
            continue
        low = str(text).strip().lower()
        for kw in keywords:
            if low.startswith(kw) and (len(low) == len(kw) or not low[len(kw)].isalpha()):
                return True
    return False


def _elements(ifc, ifc_class: str, category: str | None = None) -> list[Any]:
    try:
        items = ifc.by_type(ifc_class)
    except RuntimeError:
        return []
    if category is None:
        return list(items)
    return [e for e in items if _text_matches(e, IFC_CATEGORY_KEYWORDS[category])]


def _terrain_elements(ifc) -> list[Any]:
    out = []
    for e in _elements(ifc, "IfcGeographicElement"):
        if str(getattr(e, "PredefinedType", "") or "").upper() == "TERRAIN" or _text_matches(e, ("terrain", "gelaende", "gelände")):
            out.append(e)
    return out


def _tree_elements(ifc) -> list[Any]:
    terrain = {e.id() for e in _terrain_elements(ifc)}
    return [e for e in _elements(ifc, "IfcGeographicElement", "tree") if e.id() not in terrain]


def _body_items(element) -> list[Any]:
    """Flat list of representation items of the Body representation (mapped items resolved)."""
    items: list[Any] = []

    def walk(rep) -> None:
        for item in rep.Items or []:
            if item.is_a("IfcMappedItem"):
                walk(item.MappingSource.MappedRepresentation)
            else:
                items.append(item)

    shape = getattr(element, "Representation", None)
    if shape is None:
        return items
    for rep in shape.Representations or []:
        if rep.RepresentationIdentifier in (None, "Body"):
            walk(rep)
    return items


def _item_triangles(item) -> int:
    if item.is_a("IfcTriangulatedFaceSet"):
        return len(item.CoordIndex or [])
    if item.is_a("IfcPolygonalFaceSet"):
        return sum(max(len(f.CoordIndex) - 2, 0) for f in item.Faces or [])
    if item.is_a("IfcFaceBasedSurfaceModel") or item.is_a("IfcShellBasedSurfaceModel") or item.is_a("IfcFacetedBrep"):
        return _geom_triangles(item)
    return 0


def _geom_triangles(item) -> int:
    try:
        import ifcopenshell.geom as geom
        shape = geom.create_shape(geom.settings(), item)
        return len(shape.geometry.faces) // 3
    except Exception:  # noqa: BLE001
        return 0


def _world_mesh(element) -> tuple[np.ndarray, np.ndarray] | None:
    """Triangulated geometry of an element in world coordinates (verts N x 3, faces M x 3)."""
    try:
        import ifcopenshell.geom as geom
        settings = geom.settings()
        settings.set("use-world-coords", True)
        shape = geom.create_shape(settings, element)
        verts = np.asarray(shape.geometry.verts, dtype=float).reshape(-1, 3)
        faces = np.asarray(shape.geometry.faces, dtype=int).reshape(-1, 3)
        if len(faces) == 0:
            return None
        return verts, faces
    except Exception as error:  # noqa: BLE001
        logger.debug("no geometry for %s: %s", element, error)
        return None


def _is_closed_mesh(verts: np.ndarray, faces: np.ndarray) -> bool:
    keys = {}
    ids = [keys.setdefault(tuple(np.round(v, 5)), len(keys)) for v in verts]
    edges: dict[tuple[int, int], int] = {}
    for tri in faces:
        a, b, c = (ids[i] for i in tri)
        if len({a, b, c}) < 3:
            continue
        for u, v in ((a, b), (b, c), (c, a)):
            key = (min(u, v), max(u, v))
            edges[key] = edges.get(key, 0) + 1
    return bool(edges) and all(count == 2 for count in edges.values())


def _label_ifc(element) -> str:
    return f"{element.is_a()}#{element.id()}:{getattr(element, 'Name', None)}"


def _rule_trees(ifc, cfg: Mapping[str, Any]) -> dict[str, Any]:
    budget = cfg.get("vegetation", {}).get("crown_triangles", {"min": 60, "max": 120})
    lo, hi = int(budget["min"]), int(budget["max"])
    offenders, counts = [], []
    for tree in _tree_elements(ifc):
        n = sum(_item_triangles(i) for i in _body_items(tree) if not i.is_a("IfcExtrudedAreaSolid"))
        counts.append(n)
        if not lo <= n <= hi:
            offenders.append(f"{_label_ifc(tree)}: {n} crown triangles")
    return _result("7_tree_crown_triangles", not offenders, {"trees": len(counts), "min": min(counts, default=None),
                   "max": max(counts, default=None)}, {"min": lo, "max": hi}, offenders)


def _rule_curbs(ifc) -> dict[str, Any]:
    allowed = ("IfcExtrudedAreaSolid", "IfcFixedReferenceSweptAreaSolid")
    offenders = []
    curbs = _elements(ifc, "IfcCivilElement", "curb")
    for curb in curbs:
        bad = sorted({i.is_a() for i in _body_items(curb) if not any(i.is_a(a) for a in allowed)})
        if bad:
            offenders.append(f"{_label_ifc(curb)}: {','.join(bad)}")
    return _result("8_curb_body_types", not offenders, len(curbs), "IfcExtrudedAreaSolid / IfcFixedReferenceSweptAreaSolid", offenders)


def _pavements(ifc) -> list[Any]:
    out: list[Any] = []
    for kind in PAVEMENT_KINDS:
        out.extend(_elements(ifc, "IfcCivilElement", kind))
    return out


def _rule_pavement_thickness(ifc, cfg: Mapping[str, Any]) -> dict[str, Any]:
    min_t = float(cfg.get("pavement", {}).get("min_thickness_m", 0.10))
    offenders, thicknesses = [], []
    for element in _pavements(ifc):
        mesh = _world_mesh(element)
        if mesh is None:
            offenders.append(f"{_label_ifc(element)}: no geometry")
            continue
        verts, faces = mesh
        thickness = float(verts[:, 2].max() - verts[:, 2].min())
        thicknesses.append(thickness)
        closed = _is_closed_mesh(verts, faces)
        if thickness < min_t or not closed:
            offenders.append(f"{_label_ifc(element)}: thickness {thickness:.3f} m, closed={closed}")
    return _result("9_pavement_closed_thickness", not offenders,
                   {"pavements": len(thicknesses), "min_thickness": min(thicknesses, default=None)}, f">= {min_t} m, closed", offenders)


def _footprint_from_mesh(verts: np.ndarray, faces: np.ndarray):
    tris = []
    for tri in faces:
        p = verts[tri][:, :2]
        poly = Polygon(p)
        if poly.is_valid and poly.area > 1e-9:
            tris.append(poly)
    return unary_union(tris) if tris else None


def _triangle_stats(verts: np.ndarray, faces: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    a, b, c = verts[faces[:, 0]], verts[faces[:, 1]], verts[faces[:, 2]]
    area = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
    la, lb, lc = np.linalg.norm(b - c, axis=1), np.linalg.norm(a - c, axis=1), np.linalg.norm(a - b, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        angs = []
        for opp, s1, s2 in ((la, lb, lc), (lb, la, lc), (lc, la, lb)):
            cosv = np.clip((s1**2 + s2**2 - opp**2) / (2 * s1 * s2), -1.0, 1.0)
            angs.append(np.degrees(np.arccos(cosv)))
    min_angle = np.nan_to_num(np.min(np.stack(angs), axis=0), nan=0.0)
    return area, min_angle


def _step_faces(verts: np.ndarray, faces: np.ndarray, max_width_m: float = 0.02, min_slope_deg: float = 60.0) -> np.ndarray:
    """Triangles that are a step face: steeper than ``min_slope_deg`` and at most ``max_width_m`` wide in plan."""
    a, b, c = verts[faces[:, 0]], verts[faces[:, 1]], verts[faces[:, 2]]
    normal = np.cross(b - a, c - a)
    norm = np.maximum(np.linalg.norm(normal, axis=1), 1e-18)
    steep = np.abs(normal[:, 2]) / norm < np.cos(np.radians(min_slope_deg))
    plan_area2 = np.abs(normal[:, 2])
    longest = np.max(np.linalg.norm(np.stack((b - c, a - c, a - b))[:, :, :2], axis=2), axis=0)
    width = plan_area2 / np.maximum(longest, 1e-12)      # smallest altitude of the plan triangle
    return steep & (width <= max_width_m)


def _rule_terrain(ifc, pavement_footprints) -> dict[str, Any]:
    terrains = _terrain_elements(ifc)
    meshes = [m for m in (_world_mesh(t) for t in terrains) if m is not None]
    if not meshes:
        return _result("10_terrain_no_pavement_overlap_no_degenerate", False, "no terrain mesh", "TERRAIN mesh present", ["no IfcGeographicElement TERRAIN"])
    footprints: list[Any] = list(pavement_footprints) if pavement_footprints is not None else []
    if pavement_footprints is None:
        for element in _pavements(ifc):
            mesh = _world_mesh(element)
            fp = _footprint_from_mesh(*mesh) if mesh else None
            if fp is not None:
                footprints.append(fp)
    union = unary_union([f for f in footprints if f is not None and not f.is_empty]) if footprints else None
    from shapely import contains_xy
    offenders, inside, degenerate, total = [], 0, 0, 0
    for verts, faces in meshes:
        area, min_angle = _triangle_stats(verts, faces)
        centroids = verts[faces].mean(axis=1)
        mask_in = contains_xy(union, centroids[:, 0], centroids[:, 1]) if union is not None else np.zeros(len(faces), bool)
        mask_in &= ~_step_faces(verts, faces)   # vertical 1 cm faces of a step (samcloud/terrain/steps.py) belong to the edge, not to terrain
        mask_deg = (area < 1e-6) | (min_angle < 1.0)
        total += len(faces)
        inside += int(mask_in.sum())
        degenerate += int(mask_deg.sum())
        for idx in np.flatnonzero(mask_in)[:10]:
            offenders.append(f"triangle {idx} centroid inside pavement ({centroids[idx][0]:.2f}, {centroids[idx][1]:.2f})")
        for idx in np.flatnonzero(mask_deg)[:10]:
            offenders.append(f"triangle {idx} degenerate (area {area[idx]:.2e}, min angle {min_angle[idx]:.2f})")
    return _result("10_terrain_no_pavement_overlap_no_degenerate", inside == 0 and degenerate == 0,
                   {"triangles": total, "inside_pavement": inside, "degenerate": degenerate},
                   "0 inside, 0 with area<1e-6 m2 or angle<1 deg", offenders)


def _surface_sampler(verts: np.ndarray, faces: np.ndarray):
    """``f(xy) -> highest z of the mesh surface at xy`` (NaN where no triangle covers the point)."""
    import shapely

    a, b, c = verts[faces[:, 0]], verts[faces[:, 1]], verts[faces[:, 2]]
    tree = shapely.STRtree(shapely.polygons(verts[faces][:, :, :2])) if len(faces) else None

    def sample(xy: np.ndarray) -> np.ndarray:
        xy = np.asarray(xy, dtype=float).reshape(-1, 2)
        out = np.full(len(xy), -np.inf)
        if tree is None or not len(xy):
            return np.full(len(xy), np.nan)
        pidx, tidx = tree.query(shapely.points(xy), predicate="intersects")
        v0, v1, v2 = a[tidx], b[tidx], c[tidx]
        p = xy[pidx]
        den = (v1[:, 1] - v2[:, 1]) * (v0[:, 0] - v2[:, 0]) + (v2[:, 0] - v1[:, 0]) * (v0[:, 1] - v2[:, 1])
        ok = np.abs(den) > 1e-18
        den = np.where(ok, den, 1.0)
        w0 = ((v1[:, 1] - v2[:, 1]) * (p[:, 0] - v2[:, 0]) + (v2[:, 0] - v1[:, 0]) * (p[:, 1] - v2[:, 1])) / den
        w1 = ((v2[:, 1] - v0[:, 1]) * (p[:, 0] - v2[:, 0]) + (v0[:, 0] - v2[:, 0]) * (p[:, 1] - v2[:, 1])) / den
        z = w0 * v0[:, 2] + w1 * v1[:, 2] + (1.0 - w0 - w1) * v2[:, 2]
        np.maximum.at(out, pidx[ok], z[ok])
        out[~np.isfinite(out)] = np.nan
        return out

    return sample


def _rule_flush(ifc, cfg: Mapping[str, Any], spacing: float = 1.0, nudge: float = 0.005, inside: float = 0.03, max_bad_share: float = 0.03, max_lip: float = 0.10) -> dict[str, Any]:
    """Rule 18: along every pavement outline the terrain edge is flush with the pavement surface next to it (no lip; step faces excluded)."""
    import shapely
    from shapely.geometry.polygon import orient

    tol = float(cfg.get("terrain", {}).get("flush", {}).get("qa_tol_m", 0.01))
    terrains = [m for m in (_world_mesh(t) for t in _terrain_elements(ifc)) if m is not None]
    if not terrains:
        return _result("18_terrain_flush_at_pavement", False, "no terrain mesh", "TERRAIN mesh present", ["no terrain"])
    verts = np.vstack([v for v, _ in terrains])
    faces = np.vstack([f + sum(len(t[0]) for t in terrains[:i]) for i, (_, f) in enumerate(terrains)])
    step = _step_faces(verts, faces)
    terrain_at = _surface_sampler(verts, faces[~step])
    step_geom = unary_union([Polygon(verts[f][:, :2]) for f in faces[step] if Polygon(verts[f][:, :2]).area > 0]) if step.any() else None
    diffs: list[float] = []
    offenders: list[str] = []
    for element in _pavements(ifc):
        mesh = _world_mesh(element)
        if mesh is None:
            continue
        pv, pf = mesh
        a, b, c_ = pv[pf[:, 0]], pv[pf[:, 1]], pv[pf[:, 2]]
        normal = np.cross(b - a, c_ - a)
        top = pf[normal[:, 2] > 0.5 * np.maximum(np.linalg.norm(normal, axis=1), 1e-18)]
        footprint = _footprint_from_mesh(pv, pf)
        if footprint is None or not len(top):
            continue
        pavement_at = _surface_sampler(pv, top)
        for polygon in getattr(footprint, "geoms", [footprint]):
            polygon = orient(polygon, 1.0)
            for ring in (polygon.exterior, *polygon.interiors):
                length = ring.length
                if length < 0.5:
                    continue
                arcs = np.arange(0.5 * spacing, length, spacing)
                on = np.asarray([ring.interpolate(float(d)).coords[0][:2] for d in arcs])
                ahead = np.asarray([ring.interpolate(min(float(d) + 0.05, length)).coords[0][:2] for d in arcs])
                tangent = ahead - on
                norm = np.maximum(np.linalg.norm(tangent, axis=1), 1e-12)
                left = np.column_stack((-tangent[:, 1], tangent[:, 0])) / norm[:, None]   # inside of the oriented ring
                zp, zt = pavement_at(on + inside * left), terrain_at(on - nudge * left)   # pavement surface 3 cm in, terrain at the edge
                keep = np.isfinite(zp) & np.isfinite(zt)
                if step_geom is not None and keep.any():
                    keep &= ~np.asarray(shapely.dwithin(shapely.points(on[:, 0], on[:, 1]), step_geom, 0.05), dtype=bool)
                for point, dz in zip(on[keep], (zt - zp)[keep]):
                    diffs.append(float(dz))
                    if abs(dz) > tol and len(offenders) < MAX_OFFENDERS:
                        offenders.append(f"{_label_ifc(element)} at ({point[0]:.2f}, {point[1]:.2f}): terrain - pavement = {dz * 100:+.1f} cm")
    values = np.abs(np.asarray(diffs))
    bad = int((values > tol).sum())
    summary = {"samples": len(diffs), "over_tolerance": bad,
               "max_lip_m": round(float(values.max()), 4) if len(values) else 0.0,
               "p95_m": round(float(np.percentile(values, 95)), 4) if len(values) else 0.0}
    ok = bad <= max_bad_share * max(len(diffs), 1) and (not len(values) or float(values.max()) <= max_lip)
    return _result("18_terrain_flush_at_pavement", ok, summary,
                   f"|terrain - pavement edge| <= {tol} m on 97 % of the outline samples, never above {max_lip} m", offenders)


def _rule_georef(ifc) -> dict[str, Any]:
    import ifcopenshell.util.placement as placement
    import ifcopenshell.util.unit as unit
    problems: list[str] = []
    conv = ifc.by_type("IfcMapConversion") if "IFC4" in ifc.schema else []
    if not conv:
        problems.append("IfcMapConversion missing")
    crs_ok = False
    for crs in ifc.by_type("IfcProjectedCRS") if "IFC4" in ifc.schema else []:
        crs_ok = crs_ok or "2056" in str(crs.Name or "")
    if not crs_ok:
        problems.append("IfcProjectedCRS name does not contain 2056")
    scale = unit.calculate_unit_scale(ifc)
    far = 0
    max_dist = 0.0
    for product in ifc.by_type("IfcProduct"):
        if getattr(product, "ObjectPlacement", None) is None:
            continue
        m = placement.get_local_placement(product.ObjectPlacement)
        dist = float(np.hypot(m[0][3], m[1][3])) * scale
        max_dist = max(max_dist, dist)
        if dist > MAX_PLACEMENT_DISTANCE_M:
            far += 1
            problems.append(f"{_label_ifc(product)} placed {dist:.0f} m from origin")
    return _result("11_georeferencing", not problems, {"map_conversion": bool(conv), "crs_2056": crs_ok, "max_placement_m": round(max_dist, 1)},
                   "IfcMapConversion, EPSG 2056, placements < 10 km", problems)


class _TerrainSampler:
    """Heights of a triangulated terrain at XY positions (barycentric interpolation)."""

    def __init__(self, meshes: list[tuple[np.ndarray, np.ndarray]]) -> None:
        self.tris = np.concatenate([v[f] for v, f in meshes]) if meshes else np.zeros((0, 3, 3))

    def height(self, x: float, y: float) -> float | None:
        if len(self.tris) == 0:
            return None
        t = self.tris
        x0, y0 = t[:, 0, 0], t[:, 0, 1]
        d1x, d1y = t[:, 1, 0] - x0, t[:, 1, 1] - y0
        d2x, d2y = t[:, 2, 0] - x0, t[:, 2, 1] - y0
        den = d1x * d2y - d2x * d1y
        with np.errstate(divide="ignore", invalid="ignore"):
            u = ((x - x0) * d2y - d2x * (y - y0)) / den
            v = (d1x * (y - y0) - (x - x0) * d1y) / den
        hit = np.flatnonzero((u >= -1e-9) & (v >= -1e-9) & (u + v <= 1 + 1e-9) & np.isfinite(u))
        if len(hit) == 0:
            return None
        i = hit[0]
        return float(t[i, 0, 2] + u[i] * (t[i, 1, 2] - t[i, 0, 2]) + v[i] * (t[i, 2, 2] - t[i, 0, 2]))


def _host_wall(opening) -> Any | None:
    for rel in getattr(opening, "VoidsElements", None) or []:
        wall = rel.RelatingBuildingElement
        if wall is not None and wall.is_a("IfcWall"):
            return wall
        if wall is not None:
            return wall
    return None


def _rule_openings(ifc, cfg: Mapping[str, Any]) -> dict[str, Any]:
    tol = float(cfg.get("openings", {}).get("door_bottom_tolerance_m", 0.20))
    min_sill = float(cfg.get("openings", {}).get("window_min_sill_m", 0.40))
    sampler = _TerrainSampler([m for m in (_world_mesh(t) for t in _terrain_elements(ifc)) if m is not None])
    offenders: list[str] = []
    n_doors = n_windows = 0
    for element in _elements(ifc, "IfcDoor") + _elements(ifc, "IfcWindow"):
        is_door = element.is_a("IfcDoor")
        n_doors += is_door
        n_windows += not is_door
        fills = [r.RelatingOpeningElement for r in getattr(element, "FillsVoids", None) or []]
        host = None
        for opening in fills:
            host = host or _host_wall(opening)
        if not fills:
            offenders.append(f"{_label_ifc(element)}: fills no IfcOpeningElement")
        elif host is None:
            offenders.append(f"{_label_ifc(element)}: opening voids no wall")
        mesh = _world_mesh(element)
        if mesh is None:
            offenders.append(f"{_label_ifc(element)}: no geometry")
            continue
        verts = mesh[0]
        bottom = float(verts[:, 2].min())
        cx, cy = float(verts[:, 0].mean()), float(verts[:, 1].mean())
        ground = sampler.height(cx, cy)
        if ground is None and host is not None:
            wall_mesh = _world_mesh(host)
            ground = float(wall_mesh[0][:, 2].min()) if wall_mesh else None
        if ground is None:
            offenders.append(f"{_label_ifc(element)}: no ground reference")
            continue
        if is_door and abs(bottom - ground) > tol:
            offenders.append(f"{_label_ifc(element)}: bottom {bottom - ground:+.2f} m vs ground (tol {tol})")
        if not is_door and bottom - ground < min_sill:
            offenders.append(f"{_label_ifc(element)}: sill {bottom - ground:.2f} m < {min_sill}")
    return _result("12_openings", not offenders, {"doors": n_doors, "windows": n_windows},
                   f"door bottom within {tol} m of ground, sill >= {min_sill} m, filled voids in walls", offenders)


def _rule_ids(ifc, ids_path: Path = IDS_PATH) -> dict[str, Any]:
    try:
        from ifctester import ids as ifc_ids
        from ifctester import reporter as ids_reporter
        document = ifc_ids.open(str(ids_path), validate=True)
        document.validate(ifc)
        report = ids_reporter.Json(document).report()
    except Exception as error:  # noqa: BLE001
        return _result("13_ids", False, f"IDS check failed: {error}", "IDS pass", [str(error)])
    passed = failed = 0
    offenders: list[str] = []
    for spec in report["specifications"]:
        failed += int(spec["total_applicable_fail"])
        passed += int(spec["total_applicable"]) - int(spec["total_applicable_fail"])
        if spec["total_applicable_fail"]:
            offenders.append(f"{spec['name']}: {spec['total_applicable_fail']} of {spec['total_applicable']} failed")
    return _result("13_ids", failed == 0, {"pass": passed, "fail": failed}, "0 failures", offenders)


def check_ifc(path, config: Mapping[str, Any] | None = None, *, pavement_footprints=None) -> list[dict[str, Any]]:
    """Check an IFC file against rules 7-13; ``pavement_footprints`` (shapely polygons) override the footprints derived from pavements."""
    import ifcopenshell
    cfg = _cfg(config)
    ifc = ifcopenshell.open(str(path))
    return [
        _rule_trees(ifc, cfg), _rule_curbs(ifc), _rule_pavement_thickness(ifc, cfg),
        _rule_terrain(ifc, pavement_footprints), _rule_georef(ifc), _rule_openings(ifc, cfg), _rule_ids(ifc),
        _rule_flush(ifc, cfg),
    ]


# ============================================================================== topology


def check_topology(polygons_by_kind: Mapping[str, Sequence[Polygon]], config: Mapping[str, Any] | None = None) -> list[dict[str, Any]]:
    """Rule 14: all polygons valid; gap + overlap area of neighbouring pairs below ``pavement.max_gap_overlap_cm2``."""
    cfg = _cfg(config)
    limit = float(cfg.get("pavement", {}).get("max_gap_overlap_cm2", 1.0))
    reach = 0.05
    items = [(kind, i, p) for kind, polys in polygons_by_kind.items() for i, p in enumerate(polys)]
    invalid = [f"{k}[{i}] invalid" for k, i, p in items if not p.is_valid]
    worst = 0.0
    offenders: list[str] = []
    for a in range(len(items)):
        for b in range(a + 1, len(items)):
            ka, ia, pa = items[a]
            kb, ib, pb = items[b]
            if not (pa.is_valid and pb.is_valid) or pa.distance(pb) > 2 * reach:
                continue
            overlap = pa.intersection(pb).area
            union = unary_union([pa, pb])
            closed = union.buffer(reach, join_style=2).buffer(-reach, join_style=2)
            gap = max(closed.area - union.area, 0.0)
            cm2 = (overlap + gap) * 1e4
            worst = max(worst, cm2)
            if cm2 >= limit:
                offenders.append(f"{ka}[{ia}] / {kb}[{ib}]: overlap {overlap * 1e4:.2f} cm2, gap {gap * 1e4:.2f} cm2")
    return [
        _result("14_topology_valid", not invalid, len(items), "all polygons valid", invalid),
        _result("14_topology_gap_overlap", not offenders, round(worst, 4), f"< {limit} cm2", offenders),
    ]


# ============================================================================== count consistency


def _count_dxf(msp, category: str) -> int | None:
    layer = DXF_CATEGORY_LAYERS.get(category)
    if layer is None:
        return None
    entities = msp.query(f'*[layer=="{layer}"]')
    if category == "shrub":
        return sum(1 for e in entities if e.dxftype() == "INSERT" and e.dxf.name == "SHRUB")
    if category == "tree":  # TREE_STEM inserts (trunk circle) are no trees
        return sum(1 for e in entities if not (e.dxftype() == "INSERT" and e.dxf.name in EXTRA_BLOCKS["S-TREE"]))
    if layer in AREA_LAYERS:
        rings = [e for e in entities if _is_closed(e)]
        others = 0 if category == "vegetation_area" else len(entities) - len(rings)
        return _count_exterior_rings(rings) + others
    return len(entities)


def _ring_polygon(entity) -> Polygon | None:
    """Closed polyline -> shapely polygon (None when degenerate)."""
    if entity.dxftype() == "LWPOLYLINE":
        pts = [(float(p[0]), float(p[1])) for p in entity.get_points("xy")]
    else:
        pts = [(float(v.dxf.location[0]), float(v.dxf.location[1])) for v in entity.vertices]
    if len(pts) < 3:
        return None
    poly = Polygon(pts)
    if not poly.is_valid:
        poly = poly.buffer(0)
    return poly if isinstance(poly, Polygon) and not poly.is_empty else None


def _count_exterior_rings(rings: Sequence[Any]) -> int:
    """Count closed polylines that do not lie inside another closed polyline of the same layer (holes are not counted)."""
    polys = [_ring_polygon(e) for e in rings]
    count = 0
    for i, poly in enumerate(polys):
        if poly is None:
            count += 1
            continue
        inside = any(
            j != i and other is not None and (other.area > poly.area or (other.area == poly.area and j < i)) and other.contains(poly)
            for j, other in enumerate(polys)
        )
        count += 0 if inside else 1
    return count


def _count_ifc(ifc, category: str) -> int | None:
    if category not in IFC_CATEGORY_KEYWORDS and category not in IFC_CATEGORY_CLASSES:
        return None
    if category in ("building", "window", "door"):
        return len(_elements(ifc, IFC_CATEGORY_CLASSES[category][0]))
    terrain = {e.id() for e in _terrain_elements(ifc)}
    seen: dict[int, Any] = {}
    for cls in IFC_CATEGORY_CLASSES.get(category, IFC_DEFAULT_CLASSES):
        for e in _elements(ifc, cls, category):
            if e.id() not in terrain:
                seen[e.id()] = e
    if category == "manhole":
        for e in _elements(ifc, "IfcDistributionChamberElement"):
            if str(getattr(e, "PredefinedType", "") or "").upper() == "MANHOLE":
                seen[e.id()] = e
    return len(seen)


def count_consistency(scene_counts: Mapping[str, int], dxf_path, ifc_path) -> list[dict[str, Any]]:
    """Rule 15: per category the scene count must equal the DXF layer entity count and the IFC element count.

    ``gutter`` is its own category (``S-GUTR`` line and IFC "Wasserstein" per curb with a gutter stone). The
    geometry rules 16-17 (:func:`check_geometry_consistency`) are appended to the result list.
    """
    import ifcopenshell
    msp = ezdxf.readfile(str(dxf_path)).modelspace()
    ifc = ifcopenshell.open(str(ifc_path))
    results = []
    for category, expected in scene_counts.items():
        counts = {"scene": int(expected), "dxf": _count_dxf(msp, category), "ifc": _count_ifc(ifc, category)}
        wrong = [f"{src}={n} (scene={expected})" for src, n in counts.items() if src != "scene" and n is not None and n != expected]
        results.append(_result(f"15_count_{category}", not wrong, counts, "scene == dxf == ifc", wrong))
    return results + check_geometry_consistency(dxf_path, ifc_path)


# ============================================================================== geometry consistency (DXF vs IFC)

FENCE_LAYER = "S-FENC"
FENCE_Z_TOLERANCE_M = 0.03
FENCE_VERTEX_REACH_M = 0.08   # ribbon vertices lie within half the thickness (times the mitre factor) of the axis vertex
CURB_BODY_TOLERANCE_M = 0.005


def _dxf_points_xyz(msp, layer: str) -> np.ndarray:
    """All vertices (N,3) of the polylines on ``layer`` (LWPOLYLINE z = its elevation)."""
    rows: list[tuple[float, float, float]] = []
    for entity in msp.query(f'*[layer=="{layer}"]'):
        kind = entity.dxftype()
        if kind == "LWPOLYLINE":
            z = float(entity.dxf.elevation)
            rows += [(float(p[0]), float(p[1]), z) for p in entity.get_points("xy")]
        elif kind == "POLYLINE" and not _is_polyface(entity):
            rows += [(float(v.dxf.location.x), float(v.dxf.location.y), float(v.dxf.location.z)) for v in entity.vertices]
    return np.asarray(rows, dtype=float).reshape(-1, 3)


def _ifc_to_lv95(ifc, verts: np.ndarray) -> np.ndarray:
    """Local IFC coordinates -> LV95/LN02 with the first IfcMapConversion (identity without one)."""
    conv = ifc.by_type("IfcMapConversion") if "IFC4" in ifc.schema else []
    if not conv:
        return verts
    m = conv[0]
    a = float(m.XAxisAbscissa if m.XAxisAbscissa is not None else 1.0)
    b = float(m.XAxisOrdinate if m.XAxisOrdinate is not None else 0.0)
    scale = float(m.Scale if m.Scale is not None else 1.0)
    x, y, z = verts[:, 0], verts[:, 1], verts[:, 2]
    return np.column_stack([m.Eastings + scale * (a * x - b * y), m.Northings + scale * (b * x + a * y), m.OrthogonalHeight + scale * z])


def _rule_fence_heights(msp, ifc) -> dict[str, Any] | None:
    """The top edge of every IFC fence body equals the DXF ``S-FENC`` line (z within 3 cm)."""
    from scipy.spatial import cKDTree
    dxf = _dxf_points_xyz(msp, FENCE_LAYER)
    fences = _elements(ifc, "IfcRailing", "fence")
    if len(dxf) == 0 or not fences:
        return None
    if len(msp.query(f'*[layer=="{FENCE_LAYER}"]')) != len(fences):
        return _result("16_fence_top_ifc_vs_dxf", True, {"skipped": "S-FENC carries more lines than fences (bottom edges)"}, f"<= {FENCE_Z_TOLERANCE_M} m")
    meshes = [m for m in (_world_mesh(f) for f in fences) if m is not None]
    if not meshes:
        return None
    verts = _ifc_to_lv95(ifc, np.vstack([m[0] for m in meshes]))
    tree = cKDTree(verts[:, :2])
    offenders, worst = [], 0.0
    for p in dxf:
        near = tree.query_ball_point(p[:2], FENCE_VERTEX_REACH_M)
        if not near:
            offenders.append(f"({p[0]:.2f}, {p[1]:.2f}): no IFC fence vertex")
            continue
        # the ribbon repeats the DXF vertex as its top vertices (beside the axis by half the thickness): the best match counts
        dev = float(np.min(np.abs(verts[near, 2] - float(p[2]))))
        worst = max(worst, dev)
        if dev > FENCE_Z_TOLERANCE_M:
            offenders.append(f"({p[0]:.2f}, {p[1]:.2f}): IFC top differs by {dev:.3f} m")
    return _result("16_fence_top_ifc_vs_dxf", not offenders, {"vertices": int(len(dxf)), "max_dev_m": round(worst, 4)}, f"<= {FENCE_Z_TOLERANCE_M} m", offenders[:MAX_OFFENDERS])


def _rule_curb_body_height(msp, ifc) -> dict[str, Any] | None:
    """No IFC curb body rises above its DXF axis (``S-CURB``) by more than 5 mm."""
    from scipy.spatial import cKDTree
    dxf = _dxf_points_xyz(msp, CURB_LAYER)
    curbs = _elements(ifc, "IfcCivilElement", "curb")
    if len(dxf) == 0 or not curbs:
        return None
    tree = cKDTree(dxf[:, :2])
    offenders, checked = [], 0
    for curb in curbs:
        mesh = _world_mesh(curb)
        if mesh is None:
            continue
        verts = _ifc_to_lv95(ifc, mesh[0])
        near = {i for hits in tree.query_ball_point(verts[:, :2], 0.5) for i in hits}
        if not near:
            continue
        checked += 1
        excess = float(verts[:, 2].max()) - float(dxf[sorted(near), 2].max())
        if excess > CURB_BODY_TOLERANCE_M:
            offenders.append(f"{_label_ifc(curb)}: body {excess:.3f} m above the S-CURB axis")
    return _result("17_curb_body_below_axis", not offenders, {"curbs": checked}, f"body max z <= axis z + {CURB_BODY_TOLERANCE_M} m", offenders[:MAX_OFFENDERS])


SURFACE_FLUSH_TOLERANCE_M = 0.01
CURB_FLUSH_MAX_SLOPE = 0.3   # lateral surface gradient beyond which a curb is not judged against an upper surface


def _surface_function(ifc):
    """Height of the IFC ground surface (terrain + pavement top faces) at plan positions: ``f(xy (N,2)) -> z (N,)`` (NaN outside)."""
    import shapely
    tris = []
    for element in [*_terrain_elements(ifc), *_pavements(ifc)]:
        mesh = _world_mesh(element)
        if mesh is None:
            continue
        verts, faces = mesh
        t = verts[faces]
        normal_z = np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0])[:, 2]
        up = normal_z > 1e-9
        tris.append(t[up])
    if not tris:
        return None
    tri = np.concatenate(tris)
    tree = shapely.STRtree(shapely.polygons(np.concatenate([tri[:, :, :2], tri[:, :1, :2]], axis=1)))

    def height(xy) -> np.ndarray:
        q = np.asarray(xy, dtype=float).reshape(-1, 2)
        out = np.full(len(q), np.nan)
        qi, ti = tree.query(shapely.points(q), predicate="intersects")
        if len(qi) == 0:
            return out
        a, b, c = tri[ti, 0], tri[ti, 1], tri[ti, 2]
        d1, d2 = b[:, :2] - a[:, :2], c[:, :2] - a[:, :2]
        den = d1[:, 0] * d2[:, 1] - d2[:, 0] * d1[:, 1]
        rel = q[qi] - a[:, :2]
        with np.errstate(divide="ignore", invalid="ignore"):
            u = (rel[:, 0] * d2[:, 1] - d2[:, 0] * rel[:, 1]) / den
            w = (d1[:, 0] * rel[:, 1] - rel[:, 0] * d1[:, 1]) / den
        z = a[:, 2] + u * (b[:, 2] - a[:, 2]) + w * (c[:, 2] - a[:, 2])
        ok = np.isfinite(z)
        for k in np.flatnonzero(ok):
            out[qi[k]] = z[k] if np.isnan(out[qi[k]]) else max(out[qi[k]], z[k])
        return out

    return height


def _top_vertices(verts: np.ndarray) -> np.ndarray:
    """Highest vertex per plan position (1 mm grid): the top face of a body."""
    keys = np.round(verts[:, :2], 3)
    order = np.lexsort((-verts[:, 2], keys[:, 1], keys[:, 0]))
    first = np.r_[True, np.any(np.diff(keys[order], axis=0) != 0, axis=1)]
    return verts[order][first]


def _dxf_curb_polylines(msp) -> list[np.ndarray]:
    out = []
    for entity in msp.query(f'*[layer=="{CURB_LAYER}"]'):
        kind = entity.dxftype()
        if kind == "LWPOLYLINE":
            z = float(entity.dxf.elevation)
            pts = [(float(p[0]), float(p[1]), z) for p in entity.get_points("xy")]
        elif kind == "POLYLINE" and not _is_polyface(entity):
            pts = [(float(v.dxf.location.x), float(v.dxf.location.y), float(v.dxf.location.z)) for v in entity.vertices]
        else:
            continue
        if len(pts) >= 2:
            out.append(np.asarray(pts, dtype=float))
    return out


def _rule_gutter_curb_flush(msp, ifc) -> dict[str, Any] | None:
    """Gutter top = road surface (1 cm) and curb top = upper surface beside the curb (1 cm), against the IFC terrain/pavement surface.

    Gutter: centres of the upward faces of every gutter body against the surface at the same position. Curb: every vertex
    of the DXF ``S-CURB`` axis against the highest surface on the curb line and 1 cm to either side (the TIN step top edge lies on the line).
    """
    gutters = _elements(ifc, "IfcCivilElement", "gutter")
    curbs = _dxf_curb_polylines(msp)
    if not gutters and not curbs:
        return None
    surface = _surface_function(ifc)
    if surface is None:
        return None
    offenders: list[str] = []
    gutter_dev: list[float] = []
    p90_gutter: list[float] = []
    p90_curb: list[float] = []
    for element in gutters:
        mesh = _world_mesh(element)
        if mesh is None:
            continue
        tri = mesh[0][mesh[1]]
        up = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])[:, 2] > 1e-9
        centres = tri[up].mean(axis=1)
        dev = centres[:, 2] - surface(centres[:, :2])
        dev = np.abs(dev[np.isfinite(dev)])
        if len(dev) == 0:
            continue
        gutter_dev.append(float(np.median(dev)))
        p90_gutter.append(float(np.percentile(dev, 90)))
        if gutter_dev[-1] > SURFACE_FLUSH_TOLERANCE_M:
            offenders.append(f"{_label_ifc(element)}: gutter top differs from the surface by {gutter_dev[-1]:.3f} m (median; p90 {p90_gutter[-1]:.3f})")
    conv = ifc.by_type("IfcMapConversion") if "IFC4" in ifc.schema else []
    shift = np.array([float(conv[0].Eastings), float(conv[0].Northings), float(conv[0].OrthogonalHeight)]) if conv else np.zeros(3)
    curb_dev: list[float] = []
    for number, poly in enumerate(curbs):
        local = poly - shift
        seg = np.diff(local[:, :2], axis=0)
        seg = seg / np.maximum(np.hypot(seg[:, 0], seg[:, 1]), 1e-12)[:, None]
        tang = np.vstack([seg[:1], seg[:-1] + seg[1:], seg[-1:]])
        tang = tang / np.maximum(np.hypot(tang[:, 0], tang[:, 1]), 1e-12)[:, None]
        nrm = np.column_stack([-tang[:, 1], tang[:, 0]])
        zl, zr = surface(local[:, :2] + nrm * 0.01), surface(local[:, :2] - nrm * 0.01)
        z0 = surface(local[:, :2])
        zl_far, zr_far = surface(local[:, :2] + nrm * 0.3), surface(local[:, :2] - nrm * 0.3)
        with np.errstate(all="ignore"):
            upper = np.where(np.isfinite(z0), np.fmax(np.fmax(zl, zr), z0), np.nan)  # the curb line lies on the step top edge: the upper side wins within 1 cm; no surface under the line, no verdict
            steep = np.abs(zl_far - zr_far) / 0.6 > CURB_FLUSH_MAX_SLOPE  # on a steep slope "upper side" has no meaning
            upper = np.where(steep, np.nan, upper)
        dev = np.abs(local[:, 2] - upper)
        dev = dev[np.isfinite(dev)]
        if len(dev) == 0:
            continue
        curb_dev.append(float(np.median(dev)))
        p90_curb.append(float(np.percentile(dev, 90)))
        if curb_dev[-1] > SURFACE_FLUSH_TOLERANCE_M:
            offenders.append(f"S-CURB line {number} (LV95 {poly[0, 0]:.1f}, {poly[0, 1]:.1f}): curb top differs from the upper surface by {curb_dev[-1]:.3f} m (median; p90 {p90_curb[-1]:.3f})")
    return _result("19_gutter_curb_flush_with_surface", not offenders,
                   {"gutters": len(gutter_dev), "curbs": len(curb_dev), "max_median_gutter_m": round(max(gutter_dev, default=0.0), 4),
                    "max_median_curb_m": round(max(curb_dev, default=0.0), 4), "max_p90_gutter_m": round(max(p90_gutter, default=0.0), 4),
                    "max_p90_curb_m": round(max(p90_curb, default=0.0), 4)},
                   f"median |top - surface| <= {SURFACE_FLUSH_TOLERANCE_M} m", offenders[:MAX_OFFENDERS])


def check_geometry_consistency(dxf_path, ifc_path) -> list[dict[str, Any]]:
    """Rules 16-17 and 19: fence top edge IFC == DXF (3 cm), curb bodies never above their DXF axis (5 mm), gutter / curb tops flush with the surface (1 cm)."""
    import ifcopenshell
    msp = ezdxf.readfile(str(dxf_path)).modelspace()
    ifc = ifcopenshell.open(str(ifc_path))
    return [r for r in (_rule_fence_heights(msp, ifc), _rule_curb_body_height(msp, ifc), _rule_gutter_curb_flush(msp, ifc)) if r is not None]


# ============================================================================== report


def write_qa_report(results: Sequence[Mapping[str, Any]], out_dir) -> tuple[Path, Path]:
    """Write ``qa_cad_report.json`` and a German markdown table ``qa_cad_report.md``; return both paths."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    json_path = out / "qa_cad_report.json"
    md_path = out / "qa_cad_report.md"
    json_path.write_text(json.dumps(list(results), indent=2, ensure_ascii=False, default=str), encoding="utf-8")

    def cell(value: Any) -> str:
        text = json.dumps(value, ensure_ascii=False, default=str) if not isinstance(value, str) else value
        return text.replace("|", "/").replace("\n", " ")

    passed = sum(1 for r in results if r["ok"])
    lines = [
        "# CAD/BIM-Qualitaetspruefung", "",
        f"Ergebnis: {passed} von {len(results)} Regeln erfuellt.", "",
        "| Regel | OK | Wert | Grenze | Beispiele |", "|---|---|---|---|---|",
    ]
    for r in results:
        examples = "; ".join(r.get("offenders", [])[:3])
        lines.append(f"| {cell(r['rule'])} | {'ja' if r['ok'] else 'NEIN'} | {cell(r['value'])} | {cell(r['threshold'])} | {cell(examples)} |")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return json_path, md_path
