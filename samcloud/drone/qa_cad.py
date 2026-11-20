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
# Owner rule "Busch: Block oder geschlossene Umrisslinie mit HEIGHT": closed outlines are also allowed on these layers.
OUTLINE_ALLOWED_LAYERS = ("S-SHRUB",)
AREA_LAYERS = ("S-PAVE", "S-WALK", "S-PATH", "S-VEG")
TIN_LAYER = "S-TOPO-TIN"
CONTOUR_LAYERS = ("S-TOPO-MAJR", "S-TOPO-MINR")
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
    "fence": "S-FENC", "marking": "S-MARK", "vegetation_area": "S-VEG",
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
    """Rule 15: per category the scene count must equal the DXF layer entity count and the IFC element count."""
    import ifcopenshell
    msp = ezdxf.readfile(str(dxf_path)).modelspace()
    ifc = ifcopenshell.open(str(ifc_path))
    results = []
    for category, expected in scene_counts.items():
        counts = {"scene": int(expected), "dxf": _count_dxf(msp, category), "ifc": _count_ifc(ifc, category)}
        wrong = [f"{src}={n} (scene={expected})" for src, n in counts.items() if src != "scene" and n is not None and n != expected]
        results.append(_result(f"15_count_{category}", not wrong, counts, "scene == dxf == ifc", wrong))
    return results


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
