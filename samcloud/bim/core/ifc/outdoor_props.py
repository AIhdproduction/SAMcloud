"""Semantic attributes (Pset) and quantities (Qto) of the outdoor survey products.

One pure function per category returns ``(pset, quantities)``:

* ``pset``  - ``{property name: value}``; German names without umlauts, units in the name suffix
  (``_m``, ``_m2``, ``_m3``, ``_LN02``, ``LV95_*``).
* ``quantities`` - ``{quantity name: value}``; the suffix decides the IFC quantity type
  (``_m`` length, ``_m2`` area, ``_m3`` volume, ``Anzahl`` count).

Positions are LV95 (E/N, 3 decimals) and heights LN02: local coordinate + the georeference offset
``(dx, dy, dz)`` of the scene (``Scene.settings.origin_offset``). Unknown values are omitted, never invented;
only the explicit text "nicht gemessen" stands for a missing trunk diameter.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import shapely
from shapely.geometry import LineString, Polygon

from samcloud.bim.core.ifc.shapes import _MIN_CROWN_CLEARANCE_M, crown_mesh
from samcloud.bim.core.model import (
    BuildingElement, CurbElement, FenceElement, ManholeElement, MarkingElement, ParamOrigin,
    PavementElement, ShrubElement, SiteWallElement, TerrainElement, TreeElement, VegetationAreaElement,
)

Offset = tuple[float, float, float]
Props = tuple[dict[str, Any], dict[str, float]]

NOT_MEASURED = "nicht gemessen"
STEP_MIN_SLOPE_DEG = 60.0   # terrain triangles at least this steep count as step (Stufe)
ROOF_PLANE_ANGLE_DEG = 2.0  # roof triangles within this angle and plane distance form one roof plane
ROOF_PLANE_DISTANCE_M = 0.05

# Parameters whose origin decides the ``Source`` of a category (thicknesses and styles are conventions).
_SOURCE_KEYS: dict[str, tuple[str, ...]] = {
    "manhole": ("center", "diameter"), "tree": ("height", "crown_diameter", "trunk_xy"),
    "road": ("vertices",), "sidewalk": ("vertices",), "path": ("vertices",), "terrain": ("vertices",),
    "curb": ("axis", "foot_z"), "gutter": ("axis", "gutter_width"), "fence": ("axis", "height"),
    "site_wall": ("axis", "footprint", "top_z"), "building": ("vertices",), "roof": ("vertices",),
    "building_wall": ("vertices",), "shrub": ("height", "base_z"), "lawn": ("base_z",), "marking": ("points",),
}
_LAYER_FALLBACK = {"lawn": "S-VEG", "building": "S-BLDG"}


# ----------------------------------------------------------------------------- shared helpers


def _r3(value: float) -> float:
    return round(float(value), 3)


def _lv95(x: float, y: float, offset: Offset) -> tuple[float, float]:
    return _r3(x + offset[0]), _r3(y + offset[1])


def _ln02(z: float, offset: Offset) -> float:
    return _r3(z + offset[2])


def _layer(category: str) -> str | None:
    from samcloud.outdoor_model.style import load_outdoor_style
    style = load_outdoor_style().get(category)
    return style["layer"] if style else _LAYER_FALLBACK.get(category)


def source_of(element: Any, category: str) -> str:
    """``MEASURED`` / ``FITTED`` / ``ASSUMED`` of the geometry-defining parameters (the weakest origin wins)."""
    origins = [element.param_origin[key] for key in _SOURCE_KEYS.get(category, ()) if key in element.param_origin]
    if not origins:
        return "MEASURED"
    if any(o in (ParamOrigin.ASSUMED, ParamOrigin.TEMPLATE) for o in origins):
        return "ASSUMED"
    if any(o in (ParamOrigin.FITTED, ParamOrigin.SNAPPED) for o in origins):
        return "FITTED"
    return "MEASURED"


def _common(element: Any, category: str, offset: Offset, xy: tuple[float, float] | None, z: Sequence[float] | None,
            weights: Sequence[float] | None = None) -> dict[str, Any]:
    """Source, Layer, supporting points, RMSE, LV95 centroid and the LN02 height range (mean optionally weighted)."""
    pset: dict[str, Any] = {"Source": source_of(element, category)}
    layer = _layer(category)
    if layer:
        pset["Layer"] = layer
    points = element.metrics.point_count if element.metrics.point_count is not None else element.evidence.get("points")
    if isinstance(points, (int, float)) and not isinstance(points, bool):
        pset["Points"] = int(points)
    rmse = element.metrics.deviation_rms
    if rmse is None and isinstance(element.evidence.get("rmse"), (int, float)):
        rmse = element.evidence["rmse"]
    if rmse is not None:
        pset["RMSE_m"] = _r3(rmse)
    if xy is not None:
        pset["LV95_E"], pset["LV95_N"] = _lv95(xy[0], xy[1], offset)
    if z is not None and len(z):
        values = np.asarray(z, dtype=float)
        mean = float(np.average(values, weights=weights)) if weights is not None and float(np.sum(weights)) > 0 else float(values.mean())
        pset["Hoehe_min_LN02"], pset["Hoehe_mittel_LN02"], pset["Hoehe_max_LN02"] = (
            _ln02(values.min(), offset), _ln02(mean, offset), _ln02(values.max(), offset))
    return pset


def _polyline_length(points: Sequence[Sequence[float]]) -> float:
    xy = np.asarray([(p[0], p[1]) for p in points], dtype=float)
    return float(np.hypot(*np.diff(xy, axis=0).T).sum()) if len(xy) > 1 else 0.0


def _ring_area(ring: Sequence[Sequence[float]]) -> float:
    xy = np.asarray([(p[0], p[1]) for p in ring], dtype=float)
    x, y = xy[:, 0], xy[:, 1]
    return float(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) / 2.0)


def _ring_perimeter(ring: Sequence[Sequence[float]]) -> float:
    return _polyline_length([*ring, ring[0]])


def _mesh_stats(vertices: Sequence[Sequence[float]], triangles: Sequence[Sequence[int]]) -> dict[str, Any]:
    """Plan area, surface area, area-weighted mean slope, boundary length and height statistics of a mesh."""
    v = np.asarray(vertices, dtype=float).reshape(-1, 3)
    t = np.asarray(triangles, dtype=np.int64).reshape(-1, 3)
    if len(t) == 0 or len(v) == 0:
        return {}
    cross = np.cross(v[t[:, 1]] - v[t[:, 0]], v[t[:, 2]] - v[t[:, 0]])
    surface = 0.5 * np.linalg.norm(cross, axis=1)
    plan = 0.5 * np.abs(cross[:, 2])
    run = 0.5 * np.hypot(cross[:, 0], cross[:, 1])
    tri_z = v[t][:, :, 2].mean(axis=1)
    edge_count: dict[tuple[int, int], int] = {}
    for a, b, c in t:
        for e in ((a, b), (b, c), (c, a)):
            key = (int(min(e)), int(max(e)))
            edge_count[key] = edge_count.get(key, 0) + 1
    boundary = sum(float(np.hypot(*(v[a, :2] - v[b, :2]))) for (a, b), n in edge_count.items() if n == 1)
    plan_total = float(plan.sum())
    used = np.unique(t)
    centroid = (v[t].mean(axis=1)[:, :2] * plan[:, None]).sum(axis=0) / plan_total if plan_total > 0 else v[used, :2].mean(axis=0)
    return {
        "plan": plan_total, "surface": float(surface.sum()), "boundary": boundary, "centroid": (float(centroid[0]), float(centroid[1])),
        "slope_pct": float(run.sum() / plan_total * 100.0) if plan_total > 0 else None,
        "z_min": float(v[used, 2].min()), "z_max": float(v[used, 2].max()), "z_tri": tri_z, "weights": surface, "plan_each": plan,
        "run_each": run, "v": v, "t": t, "cross": cross,
    }


def _volume(v: np.ndarray, t: np.ndarray) -> float:
    """Absolute volume of a closed triangle mesh (divergence theorem, shifted to its centre for accuracy)."""
    p = v - v.mean(axis=0)
    return abs(float(np.einsum("ij,ij->i", p[t[:, 0]], np.cross(p[t[:, 1]], p[t[:, 2]])).sum()) / 6.0)


def _step_length(stats: Mapping[str, Any]) -> float | None:
    """Approximate length of terrain steps: half the perimeter of the union of the steep (>= 60 deg) triangles."""
    plan, run = stats["plan_each"], stats["run_each"]
    steep = (run > 0) & (plan < run * math.tan(math.radians(90.0 - STEP_MIN_SLOPE_DEG)))
    if not steep.any():
        return None
    v, t = stats["v"], stats["t"][steep]
    polygons = shapely.polygons(shapely.linearrings(v[t][:, :, :2]))
    union = shapely.union_all(polygons[shapely.area(polygons) > 0])
    return float(shapely.length(union)) / 2.0


def _surface_props(element: Any, category: str, offset: Offset, mesh: tuple[Any, Any] | None = None) -> tuple[dict[str, Any], dict[str, float], dict[str, Any]]:
    vertices, triangles = mesh if mesh is not None else (element.params.vertices, element.params.triangles)
    stats = _mesh_stats(vertices, triangles)
    if not stats:
        return _common(element, category, offset, None, None), {}, {}
    pset = _common(element, category, offset, stats["centroid"], stats["z_tri"], stats["weights"])
    pset["Hoehe_min_LN02"], pset["Hoehe_max_LN02"] = _ln02(stats["z_min"], offset), _ln02(stats["z_max"], offset)
    pset["Flaeche_m2"] = _r3(stats["plan"])
    pset["Oberflaeche_m2"] = _r3(stats["surface"])
    pset["Umfang_m"] = _r3(stats["boundary"])
    if stats["slope_pct"] is not None:
        pset["Gefaelle_mittel_%"] = round(stats["slope_pct"], 2)
    quantities = {"Flaeche_m2": pset["Flaeche_m2"], "Oberflaeche_m2": pset["Oberflaeche_m2"], "Umfang_m": pset["Umfang_m"]}
    return pset, quantities, stats


# ----------------------------------------------------------------------------- categories


def manhole_props(element: ManholeElement, offset: Offset) -> Props:
    """Cover height (same value as the DXF ``Z_DECKEL``, which is also kept as legacy ``Z_DECKEL`` in the Pset), mean height, diameter, cover area, type, roundness."""
    p = element.params
    z = float(p.center[2])
    pset = _common(element, "manhole", offset, (p.center[0], p.center[1]), [z])
    area = math.pi * (p.diameter / 2.0) ** 2
    pset.update({
        "Deckelhoehe_LN02": round(z + offset[2], 2), "Z_DECKEL": round(z + offset[2], 2), "Hoehe_mittel_LN02": _ln02(z, offset), "Durchmesser_m": _r3(p.diameter),
        "Deckelflaeche_m2": _r3(area), "Typ": p.type or "Schacht", "Deckeldicke_m": _r3(p.thickness),
    })
    for key in ("diameter_measured_m", "measured_diameter_m", "diameter_raw_m", "diameter_measured"):
        raw = element.evidence.get(key)
        if isinstance(raw, (int, float)) and not isinstance(raw, bool):
            pset["Durchmesser_gemessen_m"] = _r3(raw)
            break
    if isinstance(element.evidence.get("roundness"), (int, float)):
        pset["Rundheit"] = round(float(element.evidence["roundness"]), 3)
    quantities = {"Durchmesser_m": pset["Durchmesser_m"], "Umfang_m": _r3(math.pi * p.diameter), "Deckelflaeche_m2": pset["Deckelflaeche_m2"],
                  "Deckeldicke_m": pset["Deckeldicke_m"], "Volumen_m3": _r3(area * p.thickness), "Anzahl": 1}
    return pset, quantities


def _crown_extent(base_z: float, height: float, crown_diameter: float, crown_base_z: float | None) -> tuple[float, float]:
    """``(crown base z, crown height)`` exactly as the stylised IFC tree is built."""
    if crown_base_z is None:
        crown_height = max(height * 0.55, crown_diameter * 0.6)
        crown_base = base_z + height - crown_height
    else:
        crown_base = float(crown_base_z)
        crown_height = base_z + height - crown_base
        if crown_height < 0.2 * height or crown_height <= 0.0:
            crown_height = max(height * 0.55, crown_diameter * 0.6)
            crown_base = base_z + height - crown_height
    if crown_base < base_z + _MIN_CROWN_CLEARANCE_M:
        crown_base = base_z + _MIN_CROWN_CLEARANCE_M
        crown_height = max(base_z + height - crown_base, 0.2 * height)
    return crown_base, crown_height


def crown_volume(shape: str, crown_diameter: float, crown_height: float) -> float:
    """Volume of the stylised crown body (the closed lathe mesh of the IFC crown: sphere/ellipsoid, cone, umbrella, cylinder)."""
    vertices, triangles = crown_mesh((0.0, 0.0), 0.0, crown_height, crown_diameter, shape)
    return _volume(np.asarray(vertices, dtype=float), np.asarray(triangles, dtype=np.int64))


def tree_props(element: TreeElement, offset: Offset) -> Props:
    """Height, crown diameter/base/volume, trunk diameter ("nicht gemessen" without a measurement), positions."""
    p = element.params
    tx, ty = float(p.trunk_xy[0]), float(p.trunk_xy[1])
    cx, cy = tx + float((p.crown_offset or (0.0, 0.0))[0]), ty + float((p.crown_offset or (0.0, 0.0))[1])
    crown_base, crown_height = _crown_extent(float(p.base_z), float(p.height), float(p.crown_diameter), p.crown_base_z)
    shape = p.crown_shape or "ellipsoid"
    measured = p.trunk_measured if p.trunk_measured is not None else getattr(element.param_origin.get("trunk_diameter"), "value", "measured") != "assumed"
    top = float(p.base_z) + float(p.height)
    pset = _common(element, "tree", offset, (cx, cy), [float(p.base_z), top])
    pset["Hoehe_mittel_LN02"] = _ln02((float(p.base_z) + top) / 2.0, offset)
    volume = crown_volume(shape, float(p.crown_diameter), crown_height)
    pset.update({
        "Hoehe_m": _r3(p.height), "Kronendurchmesser_m": round(float(p.crown_diameter), 2),
        "Kronenansatz_LN02": _ln02(crown_base, offset), "Kronenhoehe_m": _r3(crown_height), "Kronenvolumen_m3": _r3(volume),
        "Fusshoehe_LN02": _ln02(float(p.base_z), offset), "StammGemessen": bool(measured),
        "Stammdurchmesser_m": _r3(p.trunk_diameter) if measured else NOT_MEASURED,
        "Kronenmitte_LV95_E": _lv95(cx, cy, offset)[0], "Kronenmitte_LV95_N": _lv95(cx, cy, offset)[1],
    })
    pset["Stamm_LV95_E"], pset["Stamm_LV95_N"] = _lv95(tx, ty, offset)
    quantities = {"Hoehe_m": pset["Hoehe_m"], "Kronendurchmesser_m": pset["Kronendurchmesser_m"], "Kronenhoehe_m": pset["Kronenhoehe_m"],
                  "Kronengrundflaeche_m2": _r3(math.pi * (float(p.crown_diameter) / 2.0) ** 2), "Kronenvolumen_m3": pset["Kronenvolumen_m3"], "Anzahl": 1}
    if measured:
        quantities["Stammdurchmesser_m"] = pset["Stammdurchmesser_m"]
    return pset, quantities


def pavement_props(element: PavementElement, offset: Offset) -> Props:
    """Plan and 3D area, perimeter, build-up thickness, volume, mean slope and height range of a road/sidewalk/path."""
    pset, quantities, stats = _surface_props(element, element.params.kind, offset)
    pset["Aufbaudicke_m"] = _r3(element.params.thickness)
    pset["Art"] = {"road": "Strasse", "sidewalk": "Trottoir", "path": "Weg"}[element.params.kind]
    quantities["Aufbaudicke_m"] = pset["Aufbaudicke_m"]
    if stats:
        pset["Volumen_m3"] = quantities["Volumen_m3"] = _r3(stats["plan"] * element.params.thickness)
    return pset, quantities


def terrain_props(element: TerrainElement, offset: Offset, mesh: tuple[Any, Any] | None = None) -> Props:
    """Plan/3D area, height range, triangle count and approximate step length of the terrain mesh as written."""
    pset, quantities, stats = _surface_props(element, "terrain", offset, mesh)
    if stats:
        pset["Dreiecke"] = int(len(stats["t"]))
        step = _step_length(stats)
        if step is not None:
            pset["Stufenlaenge_m"] = quantities["Stufenlaenge_m"] = _r3(step)
        quantities["Dreiecke"] = float(len(stats["t"]))
    return pset, quantities


def vegetation_area_props(element: VegetationAreaElement, offset: Offset, mesh: tuple[Any, Any] | None = None) -> Props:
    """Lawn area (surface mesh when it exists, else the outline), perimeter and height range."""
    if mesh is not None:
        pset, quantities, _ = _surface_props(element, "lawn", offset, mesh)
        if pset.get("Flaeche_m2") is not None:
            pset["Dreiecke"] = int(len(mesh[1]))
            return pset, quantities
    outline = element.params.outline
    centroid = np.asarray(outline, dtype=float).mean(axis=0)
    pset = _common(element, "lawn", offset, (float(centroid[0]), float(centroid[1])), [float(element.params.base_z)])
    area = _ring_area(outline)
    pset.update({"Flaeche_m2": _r3(area), "Umfang_m": _r3(_ring_perimeter(outline)), "Dreiecke": 0})
    return pset, {"Flaeche_m2": pset["Flaeche_m2"], "Umfang_m": pset["Umfang_m"]}


def shrub_props(element: ShrubElement, offset: Offset) -> Props:
    """Footprint area, perimeter, height and volume of a shrub."""
    outline, base, height = element.params.outline, float(element.params.base_z), float(element.params.height)
    centroid = np.asarray(outline, dtype=float).mean(axis=0)
    area = _ring_area(outline)
    pset = _common(element, "shrub", offset, (float(centroid[0]), float(centroid[1])), [base, base + height])
    pset.update({"Hoehe_m": _r3(height), "Grundflaeche_m2": _r3(area), "Umfang_m": _r3(_ring_perimeter(outline)),
                 "Volumen_m3": _r3(area * height), "Fusshoehe_LN02": _ln02(base, offset)})
    return pset, {k: pset[k] for k in ("Hoehe_m", "Grundflaeche_m2", "Umfang_m", "Volumen_m3")} | {"Anzahl": 1}


def curb_props(element: CurbElement, offset: Offset) -> Props:
    """Length, reveal (Anschlag), flush state, cross-section and gutter presence of a curb."""
    p = element.params
    axis = np.asarray(p.axis, dtype=float)
    length = _polyline_length(axis)
    top = axis[:, 2]
    center = LineString(axis[:, :2]).interpolate(0.5, normalized=True)
    pset = _common(element, "curb", offset, (center.x, center.y), top)
    foot = np.asarray(p.foot_z, dtype=float) if p.foot_z is not None and len(p.foot_z) == len(axis) else None
    heights = np.maximum(top - (foot - p.embed), 0.05) if foot is not None else np.full(len(axis), float(p.height))
    body = float(np.median(heights))
    pset.update({"Laenge_m": _r3(length), "Breite_m": _r3(p.width), "Hoehe_m": _r3(body), "Querschnitt": f"{p.width:.3f} x {body:.3f} m",
                 "Querschnittsflaeche_m2": round(p.width * body, 4), "Wasserstein": bool(p.gutter_width and p.gutter_width > 0),
                 "Strassenseite": {1: "links", -1: "rechts"}.get(int(p.road_side), "unbekannt")})
    quantities = {"Laenge_m": pset["Laenge_m"], "Breite_m": pset["Breite_m"], "Hoehe_m": pset["Hoehe_m"],
                  "Querschnittsflaeche_m2": pset["Querschnittsflaeche_m2"], "Volumen_m3": _r3(p.width * body * length)}
    if foot is not None:
        reveal = top - foot
        pset["Anschlag_mittel_m"] = quantities["Anschlag_mittel_m"] = _r3(float(reveal.mean()))
    elif isinstance(element.evidence.get("reveal_m"), (int, float)):
        pset["Anschlag_mittel_m"] = quantities["Anschlag_mittel_m"] = _r3(element.evidence["reveal_m"])
    if "flush" in element.evidence:
        flush = bool(element.evidence["flush"])
        pset["Buendig"] = flush
        stations = element.evidence.get("flush_stations")
        n_axis = max(len(axis), 1)
        if isinstance(stations, (int, float)) and flush is False and stations:
            pset["Buendig_Laenge_m"] = _r3(min(1.0, float(stations) / n_axis) * length)
        else:
            pset["Buendig_Laenge_m"] = _r3(length if flush else 0.0)
        quantities["Buendig_Laenge_m"] = pset["Buendig_Laenge_m"]
    if p.gutter_width and p.gutter_width > 0:
        pset["Wasserstein_Breite_m"] = _r3(p.gutter_width)
    return pset, quantities


def gutter_props(element: CurbElement, offset: Offset) -> Props:
    """Gutter stone (Wasserstein) of a curb: length, width (measured/assumed), area, thickness."""
    p = element.params
    dense = p.gutter_axis is not None and len(p.gutter_axis) >= 2
    axis = np.asarray(p.gutter_axis if dense else p.axis, dtype=float)
    length = _polyline_length(axis)
    if dense:
        top = axis[:, 2]
    elif p.foot_z is not None and len(p.foot_z) == len(axis):
        top = np.asarray(p.foot_z, dtype=float)
    else:
        top = axis[:, 2]
    center = LineString(axis[:, :2]).interpolate(0.5, normalized=True)
    pset = _common(element, "gutter", offset, (center.x, center.y), top)
    width = float(p.gutter_width)
    state = str(element.evidence.get("gutter", "")).lower()
    pset.update({"Laenge_m": _r3(length), "Breite_m": _r3(width), "Breite_Herkunft": "angenommen" if state == "assumed" else "gemessen",
                 "Flaeche_m2": _r3(length * width), "Dicke_m": _r3(p.gutter_thickness)})
    return pset, {"Laenge_m": pset["Laenge_m"], "Breite_m": pset["Breite_m"], "Flaeche_m2": pset["Flaeche_m2"],
                  "Dicke_m": pset["Dicke_m"], "Volumen_m3": _r3(length * width * p.gutter_thickness)}


def fence_props(element: FenceElement, offset: Offset) -> Props:
    """Length, height (mean/min/max) and top / bottom edge heights in LN02."""
    p = element.params
    axis = np.asarray(p.axis, dtype=float)
    bottom = np.asarray(p.bottom_z, dtype=float) if p.bottom_z is not None and len(p.bottom_z) == len(axis) else axis[:, 2] - p.height
    top = np.maximum(axis[:, 2], bottom)
    height = top - bottom
    length = _polyline_length(axis)
    center = LineString(axis[:, :2]).interpolate(0.5, normalized=True)
    pset = _common(element, "fence", offset, (center.x, center.y), top)
    pset.update({
        "Laenge_m": _r3(length), "Hoehe_mittel_m": _r3(height.mean()), "Hoehe_min_m": _r3(height.min()), "Hoehe_max_m": _r3(height.max()),
        "Oberkante_min_LN02": _ln02(top.min(), offset), "Oberkante_max_LN02": _ln02(top.max(), offset),
        "Unterkante_min_LN02": _ln02(bottom.min(), offset), "Unterkante_max_LN02": _ln02(bottom.max(), offset),
        "Dicke_m": _r3(p.thickness),
    })
    area = length * float(height.mean())
    return pset, {"Laenge_m": pset["Laenge_m"], "Hoehe_mittel_m": pset["Hoehe_mittel_m"], "Ansichtsflaeche_m2": _r3(area),
                  "Volumen_m3": _r3(area * p.thickness)}


def site_wall_props(element: SiteWallElement, offset: Offset) -> Props:
    """Length, thickness, height (mean/max), footprint area and volume of a free-standing wall."""
    from samcloud.bim.core.ifc.outdoor import _axis_footprint
    p = element.params
    thickness = float(p.thickness) if p.thickness else None
    outline = p.footprint or (_axis_footprint(p.axis, thickness) if p.axis and thickness else None)
    area = _ring_area(outline) if outline else None
    if p.axis and len(p.axis) >= 2:
        length = _polyline_length(p.axis)
    elif outline:
        length = max(_ring_perimeter(outline) / 2.0 - (thickness or 0.0), 0.0)
    else:
        length = None
    if thickness is None and area is not None and length:
        thickness = area / length
    if p.top_z_axis is not None and p.axis and len(p.top_z_axis) == len(p.axis):
        top = np.asarray(p.top_z_axis, dtype=float)
        base = np.asarray(p.base_z_axis, dtype=float) if p.base_z_axis is not None and len(p.base_z_axis) == len(top) else np.full(len(top), float(p.base_z))
    else:
        top, base = np.asarray([float(p.top_z)]), np.asarray([float(p.base_z)])
    height = np.maximum(top - base, 0.0)
    xy = np.asarray(outline if outline else p.axis, dtype=float).mean(axis=0)
    pset = _common(element, "site_wall", offset, (float(xy[0]), float(xy[1])), top)
    pset.update({"Hoehe_mittel_m": _r3(height.mean()), "Hoehe_max_m": _r3(height.max()), "Oberkante_max_LN02": _ln02(top.max(), offset),
                 "Unterkante_min_LN02": _ln02(base.min(), offset)})
    quantities: dict[str, float] = {"Hoehe_mittel_m": pset["Hoehe_mittel_m"], "Hoehe_max_m": pset["Hoehe_max_m"]}
    if length is not None:
        pset["Laenge_m"] = quantities["Laenge_m"] = _r3(length)
    if thickness is not None:
        pset["Dicke_m"] = quantities["Dicke_m"] = _r3(thickness)
    if area is not None:
        pset["Grundflaeche_m2"] = quantities["Grundflaeche_m2"] = _r3(area)
        pset["Volumen_m3"] = quantities["Volumen_m3"] = _r3(area * float(height.mean()))
    return pset, quantities


def marking_props(element: MarkingElement, offset: Offset) -> Props:
    """Length (and area of a closed marking) of a road marking."""
    points = np.asarray([(p[0], p[1], p[2] if len(p) == 3 else np.nan) for p in element.params.points], dtype=float)
    z = points[:, 2][np.isfinite(points[:, 2])]
    xy = points[:, :2].mean(axis=0)
    pset = _common(element, "marking", offset, (float(xy[0]), float(xy[1])), z if len(z) else None)
    ring = [*points[:, :2].tolist(), points[0, :2].tolist()] if element.params.closed else points[:, :2].tolist()
    pset["Laenge_m"] = _r3(_polyline_length(ring))
    pset["Geschlossen"] = bool(element.params.closed)
    quantities = {"Laenge_m": pset["Laenge_m"]}
    if element.params.closed:
        pset["Flaeche_m2"] = quantities["Flaeche_m2"] = _r3(_ring_area(points[:, :2]))
    return pset, quantities


def _roof_plane_count(stats: Mapping[str, Any], mask: np.ndarray) -> int:
    """Number of connected, coplanar roof faces (angle <= 2 deg, plane distance <= 5 cm)."""
    t, v, cross = stats["t"][mask], stats["v"], stats["cross"][mask]
    if len(t) == 0:
        return 0
    norm = np.linalg.norm(cross, axis=1)
    ok = norm > 1e-12
    normals = np.where(ok[:, None], cross / np.where(ok, norm, 1.0)[:, None], np.array([0.0, 0.0, 1.0]))
    centre = v[t].mean(axis=1)
    parent = list(range(len(t)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    edges: dict[tuple[int, int], list[int]] = {}
    for i, (a, b, c) in enumerate(t):
        for e in ((a, b), (b, c), (c, a)):
            edges.setdefault((int(min(e)), int(max(e))), []).append(i)
    cos_limit = math.cos(math.radians(ROOF_PLANE_ANGLE_DEG))
    for members in edges.values():
        for k in range(1, len(members)):
            i, j = members[0], members[k]
            if float(normals[i] @ normals[j]) >= cos_limit and abs(float(normals[i] @ (centre[j] - centre[i]))) <= ROOF_PLANE_DISTANCE_M:
                parent[find(i)] = find(j)
    return len({find(i) for i in range(len(t))})


def building_props(element: BuildingElement, offset: Offset) -> tuple[Props, Props, Props]:
    """``(building, roof, wall)`` props: footprint, facade/roof area, volume, eave/ridge heights, roof faces, RMSE, LoD."""
    p = element.params
    stats = _mesh_stats(p.vertices, p.triangles)
    if not stats:
        empty = (_common(element, "building", offset, None, None), {})
        return empty, ({}, {}), ({}, {})
    roles = np.asarray(p.face_roles, dtype=int) if p.face_roles else np.ones(len(stats["t"]), dtype=int)
    v, t = stats["v"], stats["t"]
    surface, plan = stats["weights"], stats["plan_each"]
    ground, wall, roof = roles == 0, roles == 1, roles == 2
    footprint = float(plan[ground].sum()) if ground.any() else float(plan[roof].sum())
    roof_z = v[np.unique(t[roof])][:, 2] if roof.any() else v[:, 2]
    ground_z = float(v[np.unique(t[ground])][:, 2].mean()) if ground.any() else float(v[:, 2].min())
    wall_area, roof_area = float(surface[wall].sum()), float(surface[roof].sum())
    volume = _volume(v, t)
    eaves, ridge = float(roof_z.min()), float(roof_z.max())
    base_points = v[np.unique(t[ground])][:, :2] if ground.any() else v[:, :2]
    xy = base_points.mean(axis=0)
    roof_weights = surface[roof] if roof.any() else None
    pset = _common(element, "building", offset, (float(xy[0]), float(xy[1])), stats["z_tri"][roof] if roof.any() else stats["z_tri"], roof_weights)
    pset["Hoehe_min_LN02"], pset["Hoehe_max_LN02"] = _ln02(float(v[:, 2].min()), offset), _ln02(float(v[:, 2].max()), offset)
    pset.update({"Dreiecke": int(len(t)), "Grundflaeche_m2": _r3(footprint), "Fassadenflaeche_m2": _r3(wall_area), "Dachflaeche_m2": _r3(roof_area),
                 "Volumen_m3": _r3(volume), "Traufhoehe_LN02": _ln02(eaves, offset), "Firsthoehe_LN02": _ln02(ridge, offset),
                 "Gelaendehoehe_LN02": _ln02(ground_z, offset), "Gebaeudehoehe_m": _r3(ridge - ground_z),
                 "Dachflaechen_Anzahl": _roof_plane_count(stats, roof)})
    if element.evidence.get("lod") is not None:
        pset["LoD"] = str(element.evidence["lod"])
    if isinstance(element.evidence.get("rmse"), (int, float)):
        pset["RMSE_Dach_m"] = _r3(element.evidence["rmse"])
    building_q = {k: pset[k] for k in ("Grundflaeche_m2", "Fassadenflaeche_m2", "Dachflaeche_m2", "Volumen_m3", "Gebaeudehoehe_m")}
    roof_pset = _common(element, "roof", offset, (float(xy[0]), float(xy[1])), roof_z)
    roof_pset.update({"Dachflaeche_m2": _r3(roof_area), "Traufhoehe_LN02": pset["Traufhoehe_LN02"], "Firsthoehe_LN02": pset["Firsthoehe_LN02"],
                      "Dachflaechen_Anzahl": pset["Dachflaechen_Anzahl"], "Grundflaeche_m2": _r3(float(plan[roof].sum()))})
    if "RMSE_Dach_m" in pset:
        roof_pset["RMSE_Dach_m"] = pset["RMSE_Dach_m"]
    roof_q = {"Dachflaeche_m2": roof_pset["Dachflaeche_m2"], "Grundflaeche_m2": roof_pset["Grundflaeche_m2"]}
    wall_z = v[np.unique(t[wall])][:, 2] if wall.any() else v[:, 2]
    wall_pset = _common(element, "building_wall", offset, (float(xy[0]), float(xy[1])), wall_z)
    wall_pset.update({"Fassadenflaeche_m2": _r3(wall_area), "Hoehe_max_m": _r3(float(wall_z.max() - wall_z.min())),
                      "Dreiecke": int(wall.sum())})
    wall_q = {"Fassadenflaeche_m2": wall_pset["Fassadenflaeche_m2"], "Hoehe_max_m": wall_pset["Hoehe_max_m"]}
    return (pset, building_q), (roof_pset, roof_q), (wall_pset, wall_q)


# ----------------------------------------------------------------------------- provenance

PROVENANCE_PSET = "SAMcloud_Herkunft"
PROVENANCE_KEYS = ("Source", "Points", "RMSE_m")  # moved out of the category Pset into SAMcloud_Herkunft
# Evidence entries that repeat a German category property, are local-frame values or bulk data: not copied.
_EVIDENCE_SKIP = frozenset({
    "lawn_triangle_indices", "z_deckel_local", "roundness", "lod", "rmse", "points", "views", "length_m", "area_m2", "polygon_area_m2",
    "reveal_m", "flush", "flush_stations", "gutter", "gutter_width_m", "height_m", "thickness_m", "triangle_count", "road_side",
    "face_indices", "snapped", "standard", "standard_diameter_m", "normgroesse_m", "standard_size_m", "standard_size", "diameter_measured_m",
    "measured_diameter_m", "diameter_raw_m", "diameter_measured",
})


def provenance_props(element: Any) -> dict[str, Any]:
    """Status, per-parameter origin (``Herkunft_<parameter>``), contributing views, manhole standard size / snapping and the remaining evidence."""
    props: dict[str, Any] = {"Status": element.status.value}
    for name, origin in element.param_origin.items():
        props[f"Herkunft_{name}"] = origin.value
    if element.source_instance_ids:
        props["Ansichten"] = ", ".join(element.source_instance_ids)
    if element.kind.value == "manhole":
        evidence = element.evidence
        snapped = element.param_origin.get("diameter") == ParamOrigin.SNAPPED or evidence.get("snapped") is True
        props["Eingerastet"] = bool(snapped)
        standard = evidence.get("standard")
        if isinstance(standard, str) and standard.strip():
            props["Normgroesse"] = standard.strip()
        for key in ("standard_diameter_m", "normgroesse_m", "standard_size_m", "standard_size"):
            value = evidence.get(key)
            if isinstance(value, bool):
                props["Normgroesse"] = value
                break
            if isinstance(value, (int, float)):
                props["Normgroesse_m"] = _r3(value)
                break
    manual = element.evidence.get("evidence") == "manual"  # added by the AI assistant without point-cloud evidence
    if manual:
        props["SAMcloud_Evidence"] = "manual"
    for name, value in element.evidence.items():
        if name in _EVIDENCE_SKIP or (manual and name == "evidence"):
            continue
        if isinstance(value, (str, int, float, bool)):
            props[f"Evidence_{name}"] = value
        elif isinstance(value, (list, tuple)):
            props[f"Evidence_{name}"] = ", ".join(str(item) for item in value)
        else:
            props[f"Evidence_{name}"] = str(value)
    return props


# ----------------------------------------------------------------------------- IFC typing / writing


def typed_value(file: Any, name: str, value: Any) -> Any:
    """Wrap a property value in its IFC type (IfcBoolean/IfcLabel/IfcInteger/IfcAreaMeasure/...; else the plain value)."""
    if isinstance(value, bool):
        return file.createIfcBoolean(value)
    if isinstance(value, str):
        return file.createIfcLabel(value)
    if isinstance(value, (int, np.integer)):
        return file.createIfcInteger(int(value))
    if isinstance(value, (float, np.floating)):
        if name.endswith("_m2"):
            return file.createIfcAreaMeasure(float(value))
        if name.endswith("_m3"):
            return file.createIfcVolumeMeasure(float(value))
        if name.endswith(("_m", "_LN02")) or "LV95_" in name or name.startswith("LV95"):
            return file.createIfcLengthMeasure(float(value))
        return file.createIfcReal(float(value))
    return value


def add_quantities(file: Any, product: Any, name: str, quantities: Mapping[str, float]) -> Any:
    """Attach ``IfcElementQuantity`` ``name`` (IfcQuantityLength/Area/Volume/Count by name suffix) to ``product``."""
    import ifcopenshell.api.pset
    qto = ifcopenshell.api.pset.add_qto(file, product=product, name=name)
    items = []
    for key, value in quantities.items():
        if value is None:
            continue
        if key.endswith("_m2"):
            items.append(file.createIfcQuantityArea(Name=key, AreaValue=float(value)))
        elif key.endswith("_m3"):
            items.append(file.createIfcQuantityVolume(Name=key, VolumeValue=float(value)))
        elif key.endswith("_m"):
            items.append(file.createIfcQuantityLength(Name=key, LengthValue=float(value)))
        else:
            items.append(file.createIfcQuantityCount(Name=key, CountValue=float(value)))
    qto.Quantities = items
    return qto
