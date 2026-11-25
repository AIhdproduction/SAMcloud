"""Edit operations of the outdoor assistant domain on the environment-chain Scene (local frame, metres).

:class:`OutdoorProposal` applies :class:`~samcloud.assistant.edits.EditOp` operations to a copy-on-write copy of a Scene
(the base Scene and its elements are never mutated: changed elements are replaced by new objects).  The model speaks
LV95 (``e`` / ``n``) and LN02 heights; the operations convert them with ``settings.origin_offset``.

Tier ``"elements"``: trees, manholes, fences, site walls, curbs.  Tier ``"full"`` adds pavements (delete, kind),
buildings (delete, eave / ridge height), a local terrain height change inside a polygon and
``apply_to_category`` (one operation on every element of a category that matches a filter, all or nothing).

SAMcloud computes the dependencies: tree bases, fence bottoms and wall bases are re-sampled on the terrain surface (the
terrain element of the Scene, else the final TIN), curbs are snapped to the TIN again (``snap_curb_to_surface``,
gutter recomputed), site-wall footprints rebuilt; values are validated per type.  Adding a tree / manhole needs
point-cloud evidence (canopy / trunk resp. manhole-cover points, dimensions measured); without it the op fails with
``assistant.edit.no_evidence`` unless ``manual`` is set (then ``param_origin`` assumed, ``provenance`` manual and
``evidence["evidence"] = "manual"``: IFC ``SAMcloud_Evidence = manual``, DXF layer ``<layer>-MANUAL``).

Stored ops are re-applied after a recompute by **signature** (kind + trunk / centre position, axis polyline Hausdorff
distance, main dimensions within tolerances); a target matching two elements about equally well is ambiguous.
"""

from __future__ import annotations

import copy
import math
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from samcloud.assistant.edits import CONFIRM_ELEMENT_COUNT, Change, EditError, EditOp
from samcloud.assistant.domains.outdoor_scene import EnvSupport, mesh_sampler, origin_offset

TIER_ELEMENTS = "elements"
TIER_FULL = "full"
ELEMENT_KINDS = ("tree", "manhole", "fence", "site_wall", "curb")
FULL_KINDS = ("pavement", "building", "terrain")
READ_ONLY_KINDS = ("wall", "shrub", "marking", "vegetation_area")
BULK_OP = "apply_to_category"
MANUAL_EVIDENCE = "manual"
MAX_BULK_ELEMENTS = 500

TREE_HEIGHT_RANGE_M = (1.0, 40.0)
CROWN_RANGE_M = (0.5, 25.0)
TRUNK_RANGE_M = (0.05, 3.0)
CROWN_SHAPES = ("sphere", "ellipsoid", "cone", "umbrella", "cylinder")
MANHOLE_DIAMETER_RANGE_M = (0.3, 1.5)
MANHOLE_COVER_REACH_M = 1.5  # a cover height may lie this far from the terrain
FENCE_HEIGHT_RANGE_M = (0.3, 3.0)
FENCE_THICKNESS_RANGE_M = (0.01, 0.5)
WALL_HEIGHT_RANGE_M = (0.2, 4.0)
WALL_THICKNESS_RANGE_M = (0.05, 1.5)
CURB_HEIGHT_RANGE_M = (0.01, 0.3)
CURB_WIDTH_RANGE_M = (0.05, 0.5)
GUTTER_WIDTH_RANGE_M = (0.0, 1.0)
MAX_MOVE_M = 30.0
MAX_OFFSET_M = 5.0
MAX_VERTEX_MOVE_M = 10.0
BUILDING_EAVE_RANGE_M = (2.0, 100.0)
BUILDING_RIDGE_MAX_M = 120.0
TERRAIN_DZ_MAX_M = 5.0
TERRAIN_TAPER_RANGE_M = (0.0, 5.0)
TERRAIN_DEFAULT_TAPER_M = 0.5
DUPLICATE_TREE_M = 1.0
DUPLICATE_MANHOLE_M = 0.5

SIGNATURE_POINT_TOLERANCE_M = {"tree": 1.0, "manhole": 0.5}
SIGNATURE_LINE_TOLERANCE_M = 0.75
SIGNATURE_SIZE_TOLERANCE = 0.30
SIGNATURE_AMBIGUITY_MARGIN_M = 0.05

NOT_FOUND_KEY = "assistant.edit.not_found"
INVALID_KEY = "assistant.edit.invalid_value"
BAD_PARAMS_KEY = "assistant.edit.bad_params"
NO_EVIDENCE_KEY = "assistant.edit.no_evidence"
CLOUD_REQUIRED_KEY = "assistant.edit.cloud_required"
GEOMETRY_KEY = "assistant.edit.geometry"
DUPLICATE_KEY = "assistant.outdoor.duplicate"
NO_MATCH_KEY = "assistant.outdoor.no_match"
NO_TERRAIN_KEY = "assistant.outdoor.no_terrain"
NO_ROAD_SIDE_KEY = "assistant.outdoor.no_road_side"


def _r(value: float, digits: int = 3) -> float:
    return float(round(float(value), digits))


# ----------------------------------------------------------------------------------------------------- element access

def kind_of(element: Any) -> str:
    kind = getattr(element, "kind", "")
    return str(getattr(kind, "value", kind))


def elements_of(scene: Any, kind: str) -> list[Any]:
    return [element for element in scene.elements.values() if kind_of(element) == kind]


def find_element(scene: Any, element_id: str | None) -> Any | None:
    """The element of a real id or of its short display label (``"B9"``, see ``outdoor_labels``)."""
    if element_id is None:
        return None
    element = scene.elements.get(str(element_id))
    if element is None:
        from samcloud.assistant.domains.outdoor_labels import is_label, resolve_label

        if is_label(str(element_id)):
            real = resolve_label(scene, str(element_id))
            element = scene.elements.get(real) if real is not None else None
    return element


def is_manual(element: Any) -> bool:
    return str((getattr(element, "evidence", None) or {}).get("evidence") or "") == MANUAL_EVIDENCE


def _xy(points: Any) -> np.ndarray:
    return np.asarray([(float(p[0]), float(p[1])) for p in points], dtype=np.float64).reshape(-1, 2)


def polyline_length(points: Any) -> float:
    xy = _xy(points)
    return float(np.hypot(*np.diff(xy, axis=0).T).sum()) if len(xy) > 1 else 0.0


_FOOTPRINTS: "OrderedDict[int, tuple[Any, Any]]" = OrderedDict()
_FOOTPRINT_CACHE_SIZE = 256


def mesh_footprint(element: Any) -> Any:
    """Plan footprint (shapely) of a mesh element (pavement / building / terrain); None when empty.

    Cached per element object (elements are never mutated: an edit replaces the object).
    """
    hit = _FOOTPRINTS.get(id(element))
    if hit is not None and hit[0] is element:
        return hit[1]
    footprint = _mesh_footprint(element)
    _FOOTPRINTS[id(element)] = (element, footprint)
    while len(_FOOTPRINTS) > _FOOTPRINT_CACHE_SIZE:
        _FOOTPRINTS.popitem(last=False)
    return footprint


def _mesh_footprint(element: Any) -> Any:
    import shapely
    from shapely.validation import make_valid

    vertices = np.asarray(element.params.vertices, dtype=np.float64)
    triangles = np.asarray(element.params.triangles, dtype=np.int64)
    if not len(triangles):
        return None
    polygons = shapely.polygons(vertices[triangles][:, :, :2])
    areas = shapely.area(polygons)
    geometry = make_valid(shapely.union_all(polygons[areas > 1e-12]))
    return None if geometry.is_empty else geometry


def building_heights(element: Any) -> dict[str, float]:
    """``ground`` (lowest vertex), ``eave`` (lowest roof vertex) and ``ridge`` (highest vertex) of a building (local z)."""
    vertices = np.asarray(element.params.vertices, dtype=np.float64)
    triangles = np.asarray(element.params.triangles, dtype=np.int64)
    roles = np.asarray(element.params.face_roles or [1] * len(triangles), dtype=np.int64)
    ground = float(vertices[:, 2].min())
    roof = np.unique(triangles[roles == 2].ravel()) if len(triangles) and (roles == 2).any() else np.arange(len(vertices))
    return {"ground": ground, "eave": float(vertices[roof, 2].min()), "ridge": float(vertices[:, 2].max())}


def element_height(element: Any) -> float | None:
    """Main height of an element (tree / fence / wall height, curb reveal, building ridge above ground)."""
    kind, p = kind_of(element), element.params
    if kind == "tree":
        return float(p.height)
    if kind == "fence":
        return float(p.height)
    if kind == "site_wall":
        return float(p.top_z) - float(p.base_z)
    if kind == "curb":
        axis = np.asarray(p.axis, dtype=np.float64)
        if p.foot_z is not None and len(p.foot_z) == len(axis):
            return float(np.median(axis[:, 2] - np.asarray(p.foot_z, dtype=np.float64)))
        return float(p.height)
    if kind == "building":
        heights = building_heights(element)
        return heights["ridge"] - heights["ground"]
    return None


def element_anchor(element: Any) -> tuple[float, float] | None:
    """A representative plan point (local): trunk, cover centre, polyline midpoint, footprint centroid."""
    kind, p = kind_of(element), element.params
    if kind == "tree":
        return float(p.trunk_xy[0]), float(p.trunk_xy[1])
    if kind == "manhole":
        return float(p.center[0]), float(p.center[1])
    line = _line_points(element)
    if line is not None and len(line) >= 2:
        from shapely.geometry import LineString

        mid = LineString(line).interpolate(0.5, normalized=True)
        return float(mid.x), float(mid.y)
    if getattr(p, "vertices", None):
        footprint = mesh_footprint(element)
        if footprint is not None:
            return float(footprint.centroid.x), float(footprint.centroid.y)
    for name in ("outline", "footprint", "points"):
        ring = getattr(p, name, None)
        if ring:
            xy = _xy(ring)
            return float(xy[:, 0].mean()), float(xy[:, 1].mean())
    if kind == "wall":
        return (float(p.start[0]) + float(p.end[0])) / 2.0, (float(p.start[1]) + float(p.end[1])) / 2.0
    return None


def _line_points(element: Any) -> np.ndarray | None:
    p = element.params
    axis = getattr(p, "axis", None)
    if axis:
        return _xy(axis)
    if kind_of(element) == "marking" and not p.closed:
        return _xy(p.points)
    return None


# ------------------------------------------------------------------------------------------------------- snapshots

def snapshot(scene: Any, element: Any) -> dict[str, Any]:
    """Compact JSON view of an element for change summaries (LV95 E/N, LN02 heights, metres)."""
    offset = origin_offset(scene)
    kind, p = kind_of(element), element.params

    def en(x: float, y: float) -> list[float]:
        return [_r(x + offset[0]), _r(y + offset[1])]

    def h(z: float) -> float:
        return _r(float(z) + offset[2])

    data: dict[str, Any] = {"kind": kind}
    if is_manual(element):
        data["evidence"] = MANUAL_EVIDENCE
    if kind == "tree":
        data.update({"trunk": en(*p.trunk_xy), "base_ln02": h(p.base_z), "height": _r(p.height), "crown_diameter": _r(p.crown_diameter),
                     "trunk_diameter": _r(p.trunk_diameter), "crown_shape": p.crown_shape or "ellipsoid"})
    elif kind == "manhole":
        data.update({"center": en(p.center[0], p.center[1]), "cover_ln02": h(p.center[2]), "diameter": _r(p.diameter), "type": p.type})
    elif kind in ("fence", "site_wall", "curb"):
        axis = _xy(p.axis)
        data.update({"start": en(*axis[0]), "end": en(*axis[-1]), "vertices": int(len(axis)), "length": _r(polyline_length(axis))})
        height = element_height(element)
        if height is not None:
            data["height"] = _r(height)
        if kind == "fence":
            data["thickness"] = _r(p.thickness)
        elif kind == "site_wall":
            data.update({"thickness": _r(p.thickness) if p.thickness else None, "base_ln02": h(p.base_z), "top_ln02": h(p.top_z)})
        else:
            data.update({"width": _r(p.width), "gutter_width": _r(p.gutter_width), "road_side": int(p.road_side)})
    elif kind in ("pavement", "building", "terrain"):
        footprint = mesh_footprint(element) if getattr(p, "vertices", None) else None
        if footprint is not None:
            data.update({"area_m2": _r(footprint.area, 2), "center": en(footprint.centroid.x, footprint.centroid.y)})
        if kind == "pavement":
            data["pavement_kind"] = p.kind
        elif kind == "building":
            heights = building_heights(element)
            data.update({"ground_ln02": h(heights["ground"]), "eave_ln02": h(heights["eave"]), "ridge_ln02": h(heights["ridge"])})
        else:
            z = np.asarray(p.vertices, dtype=np.float64)[:, 2] if p.vertices else np.zeros(1)
            data.update({"vertices": int(len(p.vertices)), "z_min_ln02": h(z.min()), "z_max_ln02": h(z.max())})
    else:
        anchor = element_anchor(element)
        if anchor is not None:
            data["position"] = en(*anchor)
    return data


# ------------------------------------------------------------------------------------------------------ signatures

def _simplified(points: np.ndarray, limit: int = 60) -> list[list[float]]:
    if len(points) > limit:
        index = np.unique(np.linspace(0, len(points) - 1, limit).round().astype(int))
        points = points[index]
    return [[_r(x), _r(y)] for x, y in points]


def signature_of(scene: Any, element_id: str) -> dict[str, Any] | None:
    """Kind + geometry (local metres) + main dimensions of an element, to find it again after a recompute."""
    element = find_element(scene, element_id)
    if element is None:
        return None
    kind, p = kind_of(element), element.params
    if kind == "tree":
        return {"kind": kind, "xy": [_r(p.trunk_xy[0]), _r(p.trunk_xy[1])], "height": _r(p.height), "crown_diameter": _r(p.crown_diameter)}
    if kind == "manhole":
        return {"kind": kind, "xy": [_r(p.center[0]), _r(p.center[1])], "diameter": _r(p.diameter)}
    if kind in ("fence", "site_wall", "curb"):
        axis = _xy(p.axis)
        return {"kind": kind, "axis": _simplified(axis), "length": _r(polyline_length(axis)), "height": _r(element_height(element) or 0.0)}
    if kind in ("pavement", "building"):
        footprint = mesh_footprint(element)
        if footprint is None:
            return {"kind": kind, "center": None, "area": 0.0}
        sig = {"kind": kind, "center": [_r(footprint.centroid.x), _r(footprint.centroid.y)], "area": _r(footprint.area, 2)}
        if kind == "pavement":
            sig["pavement_kind"] = p.kind
        return sig
    if kind == "terrain":
        return {"kind": kind}
    anchor = element_anchor(element)
    return {"kind": kind, "xy": None if anchor is None else [_r(anchor[0]), _r(anchor[1])]}


def _size_ok(a: float, b: float, tolerance: float = SIGNATURE_SIZE_TOLERANCE) -> bool:
    return abs(float(a) - float(b)) <= tolerance * max(abs(float(a)), abs(float(b)), 1e-6) + 0.05


def _signature_score(scene: Any, element: Any, sig: dict[str, Any]) -> float | None:
    """Distance score (lower is better) when ``element`` matches ``sig`` within the tolerances, else None."""
    current = signature_of(scene, element.id)
    kind = sig.get("kind")
    if current is None or current.get("kind") != kind:
        return None
    if kind in ("tree", "manhole"):
        distance = math.dist(sig["xy"], current["xy"])
        if distance > SIGNATURE_POINT_TOLERANCE_M[kind]:
            return None
        sizes = ("height", "crown_diameter") if kind == "tree" else ("diameter",)
        if not all(_size_ok(sig[name], current[name]) for name in sizes):
            return None
        return distance + 0.01 * sum(abs(float(sig[name]) - float(current[name])) for name in sizes)
    if kind in ("fence", "site_wall", "curb"):
        from shapely.geometry import LineString

        a, b = sig.get("axis") or [], current.get("axis") or []
        if len(a) < 2 or len(b) < 2:
            return None
        distance = float(LineString(a).hausdorff_distance(LineString(b)))
        if distance > SIGNATURE_LINE_TOLERANCE_M or not _size_ok(sig["length"], current["length"]):
            return None
        if not _size_ok(sig.get("height", 0.0), current.get("height", 0.0)):
            return None
        return distance
    if kind in ("pavement", "building"):
        if sig.get("center") is None or current.get("center") is None:
            return None
        if kind == "pavement" and sig.get("pavement_kind") != current.get("pavement_kind"):
            return None
        distance = math.dist(sig["center"], current["center"])
        if distance > max(1.0, 0.2 * math.sqrt(max(float(sig["area"]), 0.0))) or not _size_ok(sig["area"], current["area"]):
            return None
        return distance
    if kind == "terrain":
        return 0.0
    if sig.get("xy") is None or current.get("xy") is None:
        return None
    distance = math.dist(sig["xy"], current["xy"])
    return distance if distance <= 0.5 else None


def resolve_signature(scene: Any, sig: dict[str, Any] | None, preferred_id: str | None = None, *, strict: bool = False) -> str | None:
    """The id of the element matching ``sig`` best (``preferred_id`` first while it still matches), else None.

    ``strict`` (re-applying stored edits): a second candidate within :data:`SIGNATURE_AMBIGUITY_MARGIN_M` of the best
    score raises ``EditError`` ``assistant.history.ambiguous`` unless ``preferred_id`` matches exactly (unchanged).
    """
    if not sig or not sig.get("kind"):
        return None
    kind = str(sig["kind"])
    candidates = elements_of(scene, kind)
    if not strict and preferred_id is not None:
        element = find_element(scene, preferred_id)
        if element is not None and kind_of(element) == kind and _signature_score(scene, element, sig) is not None:
            return str(preferred_id)
    scored = sorted((score, element.id) for element in candidates if (score := _signature_score(scene, element, sig)) is not None)
    if not scored:
        return None
    if not strict:
        return scored[0][1]
    by_id = {element_id: score for score, element_id in scored}
    if preferred_id is not None and by_id.get(str(preferred_id), math.inf) <= 1e-6:
        return str(preferred_id)
    best_score, best = scored[0]
    rivals = [element_id for score, element_id in scored[1:] if score - best_score < SIGNATURE_AMBIGUITY_MARGIN_M]
    if rivals:
        raise EditError(f"{kind} {preferred_id or ''} matches {best} and {rivals[0]} equally well; it cannot be found again safely.",
                        key="assistant.history.ambiguous", params={"id": preferred_id, "candidates": [best, *rivals]})
    return best


def diff_scenes(base: Any, edited: Any) -> list[Change]:
    """Element changes between two Scenes that share ids: added, removed, modified."""
    changes: list[Change] = []
    for element_id, element in base.elements.items():
        other = edited.elements.get(element_id)
        if other is None:
            changes.append(Change(element_id, kind_of(element), "removed", snapshot(base, element), None))
        elif other is not element and other.model_dump(mode="json") != element.model_dump(mode="json"):
            changes.append(Change(element_id, kind_of(element), "modified", snapshot(base, element), snapshot(edited, other)))
    for element_id, element in edited.elements.items():
        if element_id not in base.elements:
            changes.append(Change(element_id, kind_of(element), "added", None, snapshot(edited, element)))
    return changes


# ------------------------------------------------------------------------------------------------------ op context

_TERRAIN_SAMPLERS: "OrderedDict[int, tuple[Any, Any]]" = OrderedDict()


def _terrain_sampler(scene: Any) -> Callable[[np.ndarray], np.ndarray] | None:
    terrains = elements_of(scene, "terrain")
    if not terrains:
        return None
    element = terrains[0]
    hit = _TERRAIN_SAMPLERS.get(id(element))
    if hit is not None and hit[0] is element:
        return hit[1]
    sampler = mesh_sampler(element.params.vertices, element.params.triangles)
    _TERRAIN_SAMPLERS[id(element)] = (element, sampler)
    while len(_TERRAIN_SAMPLERS) > 4:
        _TERRAIN_SAMPLERS.popitem(last=False)
    return sampler


def surface_heights(scene: Any, support: EnvSupport | None, xy: np.ndarray, *, tin_first: bool = False) -> np.ndarray:
    """Terrain heights (local z) at ``xy``: the Scene's terrain element (it carries terrain edits), gaps from the TIN
    (``tin_first``: the TIN first, as the curb snapping of the pipeline); NaN where neither covers."""
    xy = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
    terrain = _terrain_sampler(scene)
    first = (lambda q: support.tin_sample(q)) if (tin_first and support is not None) else terrain
    second = terrain if tin_first else ((lambda q: support.tin_sample(q)) if support is not None else None)
    z = np.full(len(xy), np.nan)
    for sampler in (first, second):
        if sampler is None:
            continue
        gap = ~np.isfinite(z)
        if not gap.any():
            break
        z[gap] = np.asarray(sampler(xy[gap]), dtype=np.float64)
    return np.where(np.isfinite(z), z, np.nan)


class _Ctx:
    def __init__(self, scene: Any, support: EnvSupport, cloud: Any, replay: bool) -> None:
        self.scene = scene
        self.support = support
        self.cloud = cloud
        self.replay = replay
        self.notes: list[str] = []
        self.resolved: dict[str, Any] = {}
        self.confirm: list[str] = []

    # elements
    def get(self, element_id: str | None, kind: str) -> Any:
        element = find_element(self.scene, element_id)
        if element is None:
            raise EditError(f"No {kind} {element_id!r} in the outdoor model.", key=NOT_FOUND_KEY, params={"id": element_id})
        if kind_of(element) != kind:
            raise EditError(f"{element_id} is a {kind_of(element)}, not a {kind}.", key=BAD_PARAMS_KEY, params={"param": "target"})
        return element

    def put(self, element: Any) -> None:
        self.scene.elements[element.id] = element

    def add(self, element: Any) -> None:
        if element.id in self.scene.elements:
            raise EditError(f"Element {element.id} exists already.", key=INVALID_KEY, params={"id": element.id})
        if getattr(element, "site_id", None) is None and self.scene.sites:
            element.site_id = next(iter(self.scene.sites))
        self.scene.elements[element.id] = element

    def drop(self, element_id: str) -> None:
        self.scene.elements.pop(element_id, None)

    # coordinates
    @property
    def offset(self) -> np.ndarray:
        return origin_offset(self.scene)

    def local_xy(self, e: float, n: float) -> tuple[float, float]:
        return float(e) - float(self.offset[0]), float(n) - float(self.offset[1])

    def surface(self, xy: np.ndarray) -> np.ndarray:
        return surface_heights(self.scene, self.support, xy)

    def curb_surface(self, xy: np.ndarray) -> np.ndarray:
        return surface_heights(self.scene, self.support, xy, tin_first=True)

    def snap(self, kind: str, element: Any, change: str = "modified") -> Change:
        return Change(element.id, kind, change, None, snapshot(self.scene, element))


def _work_copy(scene: Any) -> Any:
    """Shallow copy whose element / building dicts can be changed without touching ``scene``."""
    return scene.model_copy(update={"elements": dict(scene.elements), "buildings": dict(scene.buildings)})


def _number(params: dict[str, Any], key: str) -> float | None:
    value = params.get(key)
    if value is None:
        return None
    if isinstance(value, bool):
        raise EditError(f"{key} must be a number.", key=BAD_PARAMS_KEY, params={"param": key})
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise EditError(f"{key} must be a number.", key=BAD_PARAMS_KEY, params={"param": key}) from None
    if not math.isfinite(number):
        raise EditError(f"{key} must be finite.", key=BAD_PARAMS_KEY, params={"param": key})
    return number


def _in_range(name: str, value: float, low: float, high: float) -> float:
    if not low - 1e-9 <= value <= high + 1e-9:
        raise EditError(f"{name} {value:.3f} m is outside {low}-{high} m.", key=INVALID_KEY,
                        params={"param": name, "value": _r(value), "min": low, "max": high})
    return float(value)


def _value(params: dict[str, Any], key: str, current: float, low: float, high: float) -> float | None:
    """New value from ``key`` (absolute) or ``key_delta`` (relative to ``current``), validated; None when not given."""
    absolute, delta = _number(params, key), _number(params, f"{key}_delta")
    if absolute is not None and delta is not None:
        raise EditError(f"Give either {key} or {key}_delta.", key=BAD_PARAMS_KEY, params={"param": key})
    if absolute is None and delta is None:
        return None
    return _in_range(key, absolute if absolute is not None else float(current) + float(delta), low, high)


def _origin(element: Any, *names: str, origin: Any = None) -> dict[str, Any]:
    from samcloud.bim.core.model import ParamOrigin

    origins = dict(element.param_origin)
    for name in names:
        origins[name] = origin or ParamOrigin.MANUAL
    return origins


def _replaced(element: Any, params_update: dict[str, Any], *origin_names: str, **fields: Any) -> Any:
    params = element.params.model_copy(update=params_update)
    update = {"params": params, **fields}
    if origin_names:
        update["param_origin"] = _origin(element, *origin_names)
    return element.model_copy(update=update)


def _move_delta(ctx: _Ctx, params: dict[str, Any], current: tuple[float, float]) -> tuple[float, float]:
    """(new x, new y) local from ``dx_m`` / ``dy_m`` or an LV95 ``to`` point."""
    dx, dy = _number(params, "dx_m"), _number(params, "dy_m")
    target = params.get("to")
    if target is not None:
        if dx is not None or dy is not None:
            raise EditError("Give either to or dx_m / dy_m.", key=BAD_PARAMS_KEY, params={"param": "to"})
        x, y = _point(ctx, target, "to")
    else:
        if dx is None and dy is None:
            raise EditError("Give dx_m / dy_m or to.", key=BAD_PARAMS_KEY, params={"param": "dx_m"})
        x, y = current[0] + float(dx or 0.0), current[1] + float(dy or 0.0)
    if math.hypot(x - current[0], y - current[1]) > MAX_MOVE_M:
        raise EditError(f"Moves are limited to {MAX_MOVE_M} m.", key=INVALID_KEY, params={"param": "dx_m", "max": MAX_MOVE_M})
    return x, y


def _point(ctx: _Ctx, value: Any, name: str = "point") -> tuple[float, float]:
    if not isinstance(value, dict):
        raise EditError(f"{name} must be an object {{e, n}} (LV95).", key=BAD_PARAMS_KEY, params={"param": name})
    e, n = _number(value, "e"), _number(value, "n")
    if e is None or n is None:
        raise EditError(f"{name} needs e and n (LV95).", key=BAD_PARAMS_KEY, params={"param": name})
    return ctx.local_xy(e, n)


def _region(ctx: _Ctx, params: dict[str, Any], default_radius: float) -> Any:
    from samcloud.assistant.domains.outdoor_cloud import Region

    if params.get("box") is not None:
        box = params["box"]
        try:
            e0, e1 = sorted(float(v) for v in box["e"])
            n0, n1 = sorted(float(v) for v in box["n"])
        except (KeyError, TypeError, ValueError):
            raise EditError("box must be {e: [e0, e1], n: [n0, n1]} (LV95).", key=BAD_PARAMS_KEY, params={"param": "box"}) from None
        if e1 - e0 <= 0 or n1 - n0 <= 0 or e1 - e0 > 60 or n1 - n0 > 60:
            raise EditError("box must have a positive size of at most 60 m.", key=BAD_PARAMS_KEY, params={"param": "box"})
        x0, y0 = ctx.local_xy(e0, n0)
        x1, y1 = ctx.local_xy(e1, n1)
        return Region(box=(x0, x1, y0, y1))
    if params.get("point") is None:
        raise EditError("Give point {e, n} or box (LV95).", key=BAD_PARAMS_KEY, params={"param": "point"})
    x, y = _point(ctx, params["point"])
    radius = _number(params, "radius_m") or default_radius
    return Region(circle=(x, y, _in_range("radius_m", radius, 0.2, 15.0)))


def _new_id(kind: str) -> str:
    return f"assistant-{kind.replace('_', '-')}-{uuid.uuid4().hex[:8]}"


def _replay_added(ctx: _Ctx, op: EditOp, cls: Any, kind: str, duplicate_m: float) -> list[Change] | None:
    """Re-add the element stored with an accepted add op (``_element``); None for a fresh op."""
    data = op.params.get("_element")
    if not ctx.replay or not isinstance(data, dict):
        return None
    element = cls.model_validate(copy.deepcopy(data))
    _check_duplicate(ctx, kind, element_anchor(element), duplicate_m, ignore=element.id)
    ctx.add(element)
    return [Change(element.id, kind, "added", None, snapshot(ctx.scene, element))]


def _check_duplicate(ctx: _Ctx, kind: str, xy: tuple[float, float] | None, reach: float, ignore: str | None = None) -> None:
    if xy is None:
        return
    for other in elements_of(ctx.scene, kind):
        anchor = element_anchor(other)
        if other.id != ignore and anchor is not None and math.dist(anchor, xy) < reach:
            raise EditError(f"There is already a {kind} ({other.id}) within {reach} m.", key=DUPLICATE_KEY,
                            params={"id": other.id, "kind": kind})


def _terrain_z(ctx: _Ctx, x: float, y: float) -> float:
    z = float(ctx.surface(np.asarray([[x, y]]))[0])
    return z


# ------------------------------------------------------------------------------------------------------------ trees

def _manual_fields(provenance_manual: bool) -> dict[str, Any]:
    from samcloud.bim.core.model import Provenance

    return {"provenance": Provenance.MANUAL if provenance_manual else Provenance.AI}


def _op_add_tree(ctx: _Ctx, op: EditOp) -> list[Change]:
    from samcloud.bim.core.model import ParamOrigin, Status, TreeElement, TreeParams

    replayed = _replay_added(ctx, op, TreeElement, "tree", DUPLICATE_TREE_M)
    if replayed is not None:
        return replayed
    from samcloud.assistant.domains.outdoor_cloud import DEFAULT_TREE_RADIUS_M, MIN_TREE_POINTS, measure_tree

    params = op.params
    region = _region(ctx, params, DEFAULT_TREE_RADIUS_M)
    manual = bool(params.get("manual"))
    element_id = _new_id("tree")
    short = element_id.rsplit("-", 1)[-1][:6]
    if not manual:
        if ctx.cloud is None:
            raise EditError("Adding a tree needs the point cloud.", key=CLOUD_REQUIRED_KEY)
        measured = measure_tree(ctx.cloud, region, ctx.surface, ctx.support.config)
        if measured["points"] < MIN_TREE_POINTS or measured.get("height") is None or float(measured["height"]) < TREE_HEIGHT_RANGE_M[0]:
            raise EditError(f"No tree in the point cloud there ({measured['points']} canopy / trunk points).", key=NO_EVIDENCE_KEY,
                            params={"kind": "tree", "points": measured["points"]})
        trunk_xy = (float(measured["trunk_xy"][0]), float(measured["trunk_xy"][1]))
        _check_duplicate(ctx, "tree", trunk_xy, DUPLICATE_TREE_M)
        height = _in_range("height", float(measured["height"]), *TREE_HEIGHT_RANGE_M)
        origin = {"trunk_xy": ParamOrigin.MEASURED, "height": ParamOrigin.MEASURED,
                  "base_z": ParamOrigin.MEASURED if measured.get("terrain_found") else ParamOrigin.ASSUMED}
        crown = measured.get("crown_diameter")
        if crown is not None and CROWN_RANGE_M[0] <= float(crown) <= CROWN_RANGE_M[1]:
            origin["crown_diameter"] = ParamOrigin.MEASURED
        else:
            crown, origin["crown_diameter"] = max(1.5, height * 0.5), ParamOrigin.ASSUMED
        trunk = measured.get("trunk_diameter")
        trunk_measured = trunk is not None and float(trunk) < float(crown)
        if not trunk_measured:
            trunk = float((ctx.support.config.get("vegetation") or {}).get("default_trunk_diameter_m", 0.30))
        origin["trunk_diameter"] = ParamOrigin.MEASURED if trunk_measured else ParamOrigin.ASSUMED
        shape = str(measured.get("crown_shape") or "ellipsoid")
        origin["crown_shape"] = ParamOrigin.MEASURED if measured.get("crown_shape_status") == "measured" else ParamOrigin.ASSUMED
        offset = tuple(measured.get("crown_offset") or (0.0, 0.0))
        reach = max(float(crown) / 2.0 - float(trunk) / 2.0, 0.0)
        if math.hypot(*offset) > reach > 0:
            k = reach / math.hypot(*offset)
            offset = (offset[0] * k, offset[1] * k)
        element = TreeElement(
            id=element_id, name=f"Baum {short}",
            params=TreeParams(trunk_xy=trunk_xy, base_z=float(measured["base_z"]), height=height, crown_diameter=float(crown),
                              trunk_diameter=float(trunk), crown_shape=shape, crown_base_z=measured.get("crown_base_z"),
                              crown_offset=(float(offset[0]), float(offset[1])), trunk_measured=bool(trunk_measured)),
            param_origin=origin, status=Status.OK if trunk_measured else Status.WARNING,
            evidence={"added": "assistant", "points": int(measured["points"]), "canopy_points": int(measured["canopy_points"]),
                      "trunk_points": int(measured["trunk_points"])},
            notes=["added by the AI assistant, measured in the point cloud"], **_manual_fields(False))
    else:
        x, y = region.center
        _check_duplicate(ctx, "tree", (x, y), DUPLICATE_TREE_M)
        base = _terrain_z(ctx, x, y)
        if not math.isfinite(base):
            raise EditError("There is no terrain at this position.", key=NO_TERRAIN_KEY)
        height = _in_range("height", _number(params, "height") or 5.0, *TREE_HEIGHT_RANGE_M)
        crown = _in_range("crown_diameter", _number(params, "crown_diameter") or 3.0, *CROWN_RANGE_M)
        trunk = _in_range("trunk_diameter", _number(params, "trunk_diameter") or 0.25, *TRUNK_RANGE_M)
        shape = str(params.get("crown_shape") or "ellipsoid")
        if shape not in CROWN_SHAPES:
            raise EditError(f"crown_shape must be one of {', '.join(CROWN_SHAPES)}.", key=BAD_PARAMS_KEY, params={"param": "crown_shape"})
        element = TreeElement(
            id=element_id, name=f"Baum {short}",
            params=TreeParams(trunk_xy=(x, y), base_z=base, height=height, crown_diameter=crown, trunk_diameter=trunk,
                              crown_shape=shape, crown_offset=(0.0, 0.0), trunk_measured=False),
            # everything assumed; the crown is drawn with the given / default diameter, so DXF (KRONE) and IFC agree (QA rule 21)
            param_origin={**{name: ParamOrigin.ASSUMED for name in ("trunk_xy", "base_z", "height", "trunk_diameter", "crown_shape")},
                          "crown_diameter": ParamOrigin.MANUAL},
            status=Status.UNSURE, evidence={"evidence": MANUAL_EVIDENCE, "added": "assistant"},
            notes=["added by the AI assistant without point-cloud evidence (manual)"], **_manual_fields(True))
    ctx.add(element)
    ctx.resolved["_element"] = element.model_dump(mode="json")
    note = "manual, no point-cloud evidence" if manual else "measured in the point cloud"
    return [Change(element.id, "tree", "added", None, snapshot(ctx.scene, element), note)]


def _tree_at(ctx: _Ctx, tree: Any, x: float, y: float) -> Any:
    p = tree.params
    base = _terrain_z(ctx, x, y)
    update: dict[str, Any] = {"trunk_xy": (x, y)}
    if math.isfinite(base):
        dz = base - float(p.base_z)
        update["base_z"] = base
        if p.crown_base_z is not None:
            update["crown_base_z"] = float(p.crown_base_z) + dz
    return _replaced(tree, update, "trunk_xy")


def _op_move_tree(ctx: _Ctx, op: EditOp) -> list[Change]:
    tree = ctx.get(op.target, "tree")
    x, y = _move_delta(ctx, op.params, (float(tree.params.trunk_xy[0]), float(tree.params.trunk_xy[1])))
    moved = _tree_at(ctx, tree, x, y)
    ctx.put(moved)
    return [Change(tree.id, "tree", "moved", snapshot(ctx.scene, tree), snapshot(ctx.scene, moved))]


def _op_delete(kind: str) -> Callable[[_Ctx, EditOp], list[Change]]:
    def handler(ctx: _Ctx, op: EditOp) -> list[Change]:
        element = ctx.get(op.target, kind)
        ctx.drop(element.id)
        return [Change(element.id, kind, "removed", snapshot(ctx.scene, element), None)]

    handler.__name__ = f"_op_delete_{kind}"
    return handler


def _op_set_tree(ctx: _Ctx, op: EditOp) -> list[Change]:
    tree = ctx.get(op.target, "tree")
    p, params = tree.params, op.params
    height = _value(params, "height", p.height, *TREE_HEIGHT_RANGE_M)
    crown = _value(params, "crown_diameter", p.crown_diameter, *CROWN_RANGE_M)
    trunk = _value(params, "trunk_diameter", p.trunk_diameter, *TRUNK_RANGE_M)
    shape = params.get("crown_shape")
    if height is None and crown is None and trunk is None and shape is None:
        raise EditError("set_tree needs height, crown_diameter, trunk_diameter or crown_shape.", key=BAD_PARAMS_KEY, params={"param": "height"})
    update: dict[str, Any] = {}
    names: list[str] = []
    extra: dict[str, Any] = {}
    if height is not None:
        update["height"] = height
        names.append("height")
        if p.crown_base_z is not None and float(p.height) > 0:
            update["crown_base_z"] = float(p.base_z) + (float(p.crown_base_z) - float(p.base_z)) * height / float(p.height)
    if crown is not None:
        update["crown_diameter"] = crown
        names.append("crown_diameter")
    if trunk is not None:
        update["trunk_diameter"] = trunk
        update["trunk_measured"] = True
        names.append("trunk_diameter")
    if shape is not None:
        if str(shape) not in CROWN_SHAPES:
            raise EditError(f"crown_shape must be one of {', '.join(CROWN_SHAPES)}.", key=BAD_PARAMS_KEY, params={"param": "crown_shape"})
        update["crown_shape"] = str(shape)
        names.append("crown_shape")
    new_crown = float(update.get("crown_diameter", p.crown_diameter))
    new_trunk = float(update.get("trunk_diameter", p.trunk_diameter))
    if new_trunk >= new_crown:
        raise EditError("The trunk must be thinner than the crown.", key=INVALID_KEY, params={"param": "trunk_diameter"})
    offset = tuple(p.crown_offset or (0.0, 0.0))
    reach = max(new_crown / 2.0 - new_trunk / 2.0, 0.0)
    if math.hypot(*offset) > reach:
        k = reach / math.hypot(*offset) if math.hypot(*offset) > 0 else 0.0
        update["crown_offset"] = (float(offset[0]) * k, float(offset[1]) * k)
    if trunk is not None:
        extra["notes"] = [n for n in tree.notes if n != "assumed: trunk_diameter"]
    changed = _replaced(tree, update, *names, **extra)
    ctx.put(changed)
    return [Change(tree.id, "tree", "modified", snapshot(ctx.scene, tree), snapshot(ctx.scene, changed))]


# --------------------------------------------------------------------------------------------------------- manholes

def _op_add_manhole(ctx: _Ctx, op: EditOp) -> list[Change]:
    from samcloud.bim.core.model import ManholeElement, ManholeParams, ParamOrigin, Status

    replayed = _replay_added(ctx, op, ManholeElement, "manhole", DUPLICATE_MANHOLE_M)
    if replayed is not None:
        return replayed
    from samcloud.assistant.domains.outdoor_cloud import DEFAULT_MANHOLE_RADIUS_M, MIN_MANHOLE_POINTS, measure_manhole

    params = op.params
    region = _region(ctx, params, DEFAULT_MANHOLE_RADIUS_M)
    manual = bool(params.get("manual"))
    element_id = _new_id("manhole")
    short = element_id.rsplit("-", 1)[-1][:6]
    type_name = str(params.get("type") or "Schacht")[:40]
    if not manual:
        if ctx.cloud is None:
            raise EditError("Adding a manhole needs the point cloud.", key=CLOUD_REQUIRED_KEY)
        measured = measure_manhole(ctx.cloud, region, ctx.support.config)
        if measured["points"] < MIN_MANHOLE_POINTS or measured.get("center") is None:
            raise EditError(f"No manhole cover in the point cloud there ({measured['points']} cover points).", key=NO_EVIDENCE_KEY,
                            params={"kind": "manhole", "points": measured["points"]})
        cx, cy, cz = (float(v) for v in measured["center"])
        _check_duplicate(ctx, "manhole", (cx, cy), DUPLICATE_MANHOLE_M)
        diameter = _in_range("diameter", float(measured["diameter"]), *MANHOLE_DIAMETER_RANGE_M)
        element = ManholeElement(
            id=element_id, tag=short, name=f"Schacht {short}",
            params=ManholeParams(center=(cx, cy, cz), diameter=diameter, type=type_name),
            param_origin={"center": ParamOrigin.MEASURED, "diameter": ParamOrigin.SNAPPED if measured.get("snapped") else ParamOrigin.FITTED,
                          "type": ParamOrigin.ASSUMED},
            evidence={"added": "assistant", "points": int(measured["points"]), "z_deckel_local": round(cz, 3),
                      "diameter_measured_m": measured.get("diameter_measured"), "snapped": bool(measured.get("snapped")),
                      "standard": str(measured.get("standard") or "")},
            status=Status.OK, notes=["added by the AI assistant, measured in the point cloud"], **_manual_fields(False))
    else:
        cx, cy = region.center
        _check_duplicate(ctx, "manhole", (cx, cy), DUPLICATE_MANHOLE_M)
        cz = _terrain_z(ctx, cx, cy)
        if not math.isfinite(cz):
            raise EditError("There is no terrain at this position.", key=NO_TERRAIN_KEY)
        diameter = _in_range("diameter", _number(params, "diameter") or 0.6, *MANHOLE_DIAMETER_RANGE_M)
        element = ManholeElement(
            id=element_id, tag=short, name=f"Schacht {short}",
            params=ManholeParams(center=(cx, cy, cz), diameter=diameter, type=type_name),
            param_origin={"center": ParamOrigin.ASSUMED, "diameter": ParamOrigin.ASSUMED, "type": ParamOrigin.ASSUMED},
            evidence={"evidence": MANUAL_EVIDENCE, "added": "assistant"}, status=Status.UNSURE,
            notes=["added by the AI assistant without point-cloud evidence (manual)"], **_manual_fields(True))
    ctx.add(element)
    ctx.resolved["_element"] = element.model_dump(mode="json")
    note = "manual, no point-cloud evidence" if manual else "measured in the point cloud"
    return [Change(element.id, "manhole", "added", None, snapshot(ctx.scene, element), note)]


def _op_move_manhole(ctx: _Ctx, op: EditOp) -> list[Change]:
    manhole = ctx.get(op.target, "manhole")
    cx, cy, cz = (float(v) for v in manhole.params.center)
    x, y = _move_delta(ctx, op.params, (cx, cy))
    old_ground, new_ground = _terrain_z(ctx, cx, cy), _terrain_z(ctx, x, y)
    z = cz + (new_ground - old_ground) if math.isfinite(old_ground) and math.isfinite(new_ground) else cz
    moved = _replaced(manhole, {"center": (x, y, z)}, "center")
    ctx.put(moved)
    return [Change(manhole.id, "manhole", "moved", snapshot(ctx.scene, manhole), snapshot(ctx.scene, moved))]


def _op_set_manhole(ctx: _Ctx, op: EditOp) -> list[Change]:
    manhole = ctx.get(op.target, "manhole")
    p, params = manhole.params, op.params
    diameter = _value(params, "diameter", p.diameter, *MANHOLE_DIAMETER_RANGE_M)
    cover, cover_delta = _number(params, "cover_z"), _number(params, "cover_z_delta")
    type_name = params.get("type")
    if diameter is None and cover is None and cover_delta is None and type_name is None:
        raise EditError("set_manhole needs diameter, cover_z or type.", key=BAD_PARAMS_KEY, params={"param": "diameter"})
    update: dict[str, Any] = {}
    names: list[str] = []
    if diameter is not None:
        update["diameter"] = diameter
        names.append("diameter")
    if cover is not None or cover_delta is not None:
        if cover is not None and cover_delta is not None:
            raise EditError("Give either cover_z or cover_z_delta.", key=BAD_PARAMS_KEY, params={"param": "cover_z"})
        z = float(cover) - float(ctx.offset[2]) if cover is not None else float(p.center[2]) + float(cover_delta)
        ground = _terrain_z(ctx, float(p.center[0]), float(p.center[1]))
        if math.isfinite(ground) and abs(z - ground) > MANHOLE_COVER_REACH_M:
            raise EditError(f"The cover would lie {z - ground:+.2f} m from the terrain (max {MANHOLE_COVER_REACH_M} m).", key=INVALID_KEY,
                            params={"param": "cover_z", "max": MANHOLE_COVER_REACH_M})
        update["center"] = (float(p.center[0]), float(p.center[1]), z)
        names.append("center")
    if type_name is not None:
        text = str(type_name).strip()
        if not 1 <= len(text) <= 40:
            raise EditError("type must have 1-40 characters.", key=BAD_PARAMS_KEY, params={"param": "type"})
        update["type"] = text
        names.append("type")
    changed = _replaced(manhole, update, *names)
    ctx.put(changed)
    return [Change(manhole.id, "manhole", "modified", snapshot(ctx.scene, manhole), snapshot(ctx.scene, changed))]


# --------------------------------------------------------------------------------------------------- line elements

def offset_polyline(xy: np.ndarray, distance: float) -> np.ndarray:
    """Parallel polyline at ``distance`` (positive = left of the direction first -> last vertex), mitre joins."""
    xy = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
    seg = np.diff(xy, axis=0)
    length = np.hypot(seg[:, 0], seg[:, 1])
    if (length <= 1e-9).any():
        keep = np.r_[True, length > 1e-9]
        xy = xy[keep]
        seg = np.diff(xy, axis=0)
        length = np.hypot(seg[:, 0], seg[:, 1])
    if len(xy) < 2:
        raise EditError("The axis is degenerate.", key=GEOMETRY_KEY)
    normals = np.column_stack([-seg[:, 1], seg[:, 0]]) / length[:, None]
    out = np.empty_like(xy)
    out[0] = xy[0] + normals[0] * distance
    out[-1] = xy[-1] + normals[-1] * distance
    for i in range(1, len(xy) - 1):
        bisector = normals[i - 1] + normals[i]
        norm = float(np.hypot(*bisector))
        if norm < 1e-9:
            out[i] = xy[i] + normals[i] * distance
            continue
        bisector /= norm
        cos_half = max(float(bisector @ normals[i]), 0.25)
        out[i] = xy[i] + bisector * distance / cos_half
    return out


def _axis_ground(ctx: _Ctx, xy: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    z = ctx.surface(xy)
    return np.where(np.isfinite(z), z, fallback)


def _fence_with_axis(ctx: _Ctx, fence: Any, xy: np.ndarray) -> Any:
    p = fence.params
    axis = np.asarray(p.axis, dtype=np.float64)
    if p.bottom_z is not None and len(p.bottom_z) == len(axis):
        old_bottom = np.asarray(p.bottom_z, dtype=np.float64)
    else:
        old_bottom = axis[:, 2] - float(p.height)
    heights = axis[:, 2] - old_bottom
    bottom = _axis_ground(ctx, xy, old_bottom)
    top = bottom + heights
    return _replaced(fence, {"axis": [(float(x), float(y), float(z)) for (x, y), z in zip(xy, top)],
                             "bottom_z": [float(v) for v in bottom]}, "axis")


def _wall_with_axis(ctx: _Ctx, wall: Any, xy: np.ndarray, thickness: float | None = None) -> Any:
    from samcloud.outdoor_model.site_walls import wall_polygon

    p = wall.params
    if p.axis is None:
        raise EditError(f"Site wall {wall.id} has no axis (footprint only); it can only be deleted or changed in height.",
                        key=GEOMETRY_KEY, params={"id": wall.id})
    thickness = float(thickness if thickness is not None else p.thickness)
    n = len(xy)
    old_n = len(p.axis)
    base_axis = np.asarray(p.base_z_axis, dtype=np.float64) if p.base_z_axis is not None and len(p.base_z_axis) == old_n else np.full(old_n, float(p.base_z))
    top_axis = np.asarray(p.top_z_axis, dtype=np.float64) if p.top_z_axis is not None and len(p.top_z_axis) == old_n else np.full(old_n, float(p.top_z))
    if n != old_n:
        raise EditError("The axis vertex count cannot change.", key=GEOMETRY_KEY)
    heights = top_axis - base_axis
    ground = _axis_ground(ctx, xy, base_axis)
    update: dict[str, Any] = {"axis": [(float(x), float(y)) for x, y in xy], "thickness": thickness}
    if p.base_z_axis is not None or p.top_z_axis is not None:
        update["base_z_axis"] = [float(v) for v in ground]
        update["top_z_axis"] = [float(v) for v in ground + heights]
        update["base_z"] = float(ground.min())
        update["top_z"] = float((ground + heights).max())
    else:
        base = float(np.median(ground))
        update["base_z"] = base
        update["top_z"] = base + float(np.median(heights))
    try:
        ring = wall_polygon(xy, thickness)
    except Exception as error:  # noqa: BLE001 - a degenerate axis gives no footprint
        raise EditError(f"The site wall footprint is degenerate ({type(error).__name__}).", key=GEOMETRY_KEY) from None
    update["footprint"] = [(float(x), float(y)) for x, y in ring]
    return _replaced(wall, update, "axis")


def _op_set_fence(ctx: _Ctx, op: EditOp) -> list[Change]:
    fence = ctx.get(op.target, "fence")
    p, params = fence.params, op.params
    height = _value(params, "height", p.height, *FENCE_HEIGHT_RANGE_M)
    thickness = _value(params, "thickness", p.thickness, *FENCE_THICKNESS_RANGE_M)
    if height is None and thickness is None:
        raise EditError("set_fence needs height or thickness.", key=BAD_PARAMS_KEY, params={"param": "height"})
    update: dict[str, Any] = {}
    names: list[str] = []
    if height is not None:
        axis = np.asarray(p.axis, dtype=np.float64)
        bottom = np.asarray(p.bottom_z, dtype=np.float64) if p.bottom_z is not None and len(p.bottom_z) == len(axis) else axis[:, 2] - float(p.height)
        update["axis"] = [(float(x), float(y), float(b + height)) for (x, y, _z), b in zip(axis, bottom)]
        update["bottom_z"] = [float(v) for v in bottom]
        update["height"] = height
        names += ["height", "axis"]
    if thickness is not None:
        update["thickness"] = thickness
        names.append("thickness")
    evidence = dict(fence.evidence)
    if height is not None and "height_m" in evidence:
        evidence["height_m"] = _r(height)
    changed = _replaced(fence, update, *names, evidence=evidence)
    ctx.put(changed)
    return [Change(fence.id, "fence", "modified", snapshot(ctx.scene, fence), snapshot(ctx.scene, changed))]


def _op_offset_line(kind: str) -> Callable[[_Ctx, EditOp], list[Change]]:
    def handler(ctx: _Ctx, op: EditOp) -> list[Change]:
        element = ctx.get(op.target, kind)
        distance = _number(op.params, "distance_m")
        if distance is None or abs(distance) < 1e-4:
            raise EditError("distance_m is missing or zero.", key=BAD_PARAMS_KEY, params={"param": "distance_m"})
        _in_range("distance_m", abs(distance), 0.0, MAX_OFFSET_M)
        if element.params.axis is None:
            raise EditError(f"{element.id} has no axis.", key=GEOMETRY_KEY, params={"id": element.id})
        xy = offset_polyline(_xy(element.params.axis), distance)
        changed = _with_axis(ctx, kind, element, xy)
        ctx.put(changed)
        return [Change(element.id, kind, "moved", snapshot(ctx.scene, element), snapshot(ctx.scene, changed))]

    handler.__name__ = f"_op_offset_{kind}"
    return handler


def _op_move_vertex(kind: str) -> Callable[[_Ctx, EditOp], list[Change]]:
    def handler(ctx: _Ctx, op: EditOp) -> list[Change]:
        element = ctx.get(op.target, kind)
        if element.params.axis is None:
            raise EditError(f"{element.id} has no axis.", key=GEOMETRY_KEY, params={"id": element.id})
        xy = _xy(element.params.axis)
        index = op.params.get("index")
        if isinstance(index, bool) or not isinstance(index, (int, float)) or int(index) != index:
            raise EditError("index must be an integer.", key=BAD_PARAMS_KEY, params={"param": "index"})
        index = int(index)
        if index < 0:
            index += len(xy)
        if not 0 <= index < len(xy):
            raise EditError(f"index must be 0-{len(xy) - 1}.", key=BAD_PARAMS_KEY, params={"param": "index"})
        dx, dy = _number(op.params, "dx_m") or 0.0, _number(op.params, "dy_m") or 0.0
        if math.hypot(dx, dy) < 1e-4 or math.hypot(dx, dy) > MAX_VERTEX_MOVE_M:
            raise EditError(f"The vertex move must be between 0 and {MAX_VERTEX_MOVE_M} m.", key=INVALID_KEY, params={"param": "dx_m"})
        xy = xy.copy()
        xy[index] += (dx, dy)
        seg = np.hypot(*np.diff(xy, axis=0).T)
        if (seg < 0.05).any():
            raise EditError("The vertex would collapse a segment.", key=GEOMETRY_KEY)
        changed = _with_axis(ctx, kind, element, xy)
        ctx.put(changed)
        return [Change(element.id, kind, "modified", snapshot(ctx.scene, element), snapshot(ctx.scene, changed))]

    handler.__name__ = f"_op_move_{kind}_vertex"
    return handler


def _with_axis(ctx: _Ctx, kind: str, element: Any, xy: np.ndarray) -> Any:
    if kind == "fence":
        return _fence_with_axis(ctx, element, xy)
    if kind == "site_wall":
        return _wall_with_axis(ctx, element, xy)
    return _curb_with_axis(ctx, element, xy)


def _op_set_site_wall(ctx: _Ctx, op: EditOp) -> list[Change]:
    wall = ctx.get(op.target, "site_wall")
    p, params = wall.params, op.params
    current_height = float(p.top_z) - float(p.base_z)
    if p.top_z_axis is not None and p.base_z_axis is not None and len(p.top_z_axis) == len(p.base_z_axis):
        current_height = float(np.median(np.asarray(p.top_z_axis) - np.asarray(p.base_z_axis)))
    height = _value(params, "height", current_height, *WALL_HEIGHT_RANGE_M)
    thickness = _value(params, "thickness", float(p.thickness or 0.2), *WALL_THICKNESS_RANGE_M)
    if height is None and thickness is None:
        raise EditError("set_site_wall needs height or thickness.", key=BAD_PARAMS_KEY, params={"param": "height"})
    changed = wall
    names: list[str] = []
    if thickness is not None:
        if p.axis is None:
            raise EditError(f"Site wall {wall.id} has no axis; its thickness cannot change.", key=GEOMETRY_KEY, params={"id": wall.id})
        from samcloud.outdoor_model.site_walls import wall_polygon

        ring = wall_polygon(_xy(p.axis), thickness)
        changed = wall.model_copy(update={"params": p.model_copy(update={
            "thickness": thickness, "footprint": [(float(x), float(y)) for x, y in ring]})})
        names += ["thickness", "footprint"]
    if height is not None:
        q = changed.params
        update: dict[str, Any] = {"top_z": float(q.base_z) + height}
        if q.base_z_axis is not None:
            tops = [float(b) + height for b in q.base_z_axis]
            update["top_z_axis"] = tops
            update["top_z"] = max(tops)
        elif q.top_z_axis is not None:
            update["top_z_axis"] = [float(q.base_z) + height] * len(q.top_z_axis)
        changed = changed.model_copy(update={"params": q.model_copy(update=update)})
        names.append("top_z")
    changed = changed.model_copy(update={"param_origin": _origin(changed, *names)})
    ctx.put(changed)
    return [Change(wall.id, "site_wall", "modified", snapshot(ctx.scene, wall), snapshot(ctx.scene, changed))]


# ------------------------------------------------------------------------------------------------------------ curbs

def _curb_cc(ctx: _Ctx, width: float, gutter_width: float) -> dict[str, Any]:
    cc = dict((ctx.support.config or {}).get("curb", {}) or {})
    cc["section_width_m"] = float(width)
    if gutter_width > 0:
        cc["gutter_width_m"] = float(gutter_width)
    return cc


def _snap(ctx: _Ctx, pts: np.ndarray, road_side: int, width: float, gutter_width: float) -> dict[str, Any] | None:
    from samcloud.outdoor_model.civil import snap_curb_to_surface

    steps = ((ctx.support.config or {}).get("terrain", {}) or {}).get("step", {}) or {}
    try:
        return snap_curb_to_surface(pts, int(road_side), float(gutter_width), _curb_cc(ctx, width, gutter_width), ctx.curb_surface,
                                    ctx.support.paved, ctx.support.outline, steps)
    except Exception as error:  # noqa: BLE001 - without a snap the curb keeps its heights
        ctx.notes.append(f"curb not snapped to the terrain ({type(error).__name__})")
        return None


def _gutter_update(snapped: dict[str, Any] | None, gutter_width: float) -> dict[str, Any]:
    if gutter_width <= 0:
        return {"gutter_width": 0.0, "gutter_axis": None, "gutter_z_outer": None}
    if snapped is not None and snapped.get("gutter") and snapped.get("gutter_axis") is not None:
        return {"gutter_width": float(gutter_width), "gutter_axis": [tuple(map(float, q)) for q in snapped["gutter_axis"]],
                "gutter_z_outer": [float(v) for v in snapped["gutter_z_outer"]]}
    return {"gutter_width": float(gutter_width), "gutter_axis": None, "gutter_z_outer": None}


def _curb_with_axis(ctx: _Ctx, curb: Any, xy: np.ndarray) -> Any:
    p = curb.params
    axis = np.asarray(p.axis, dtype=np.float64)
    if len(xy) != len(axis):
        raise EditError("The axis vertex count cannot change.", key=GEOMETRY_KEY)
    old_ground = _axis_ground(ctx, axis[:, :2], axis[:, 2])
    top = _axis_ground(ctx, xy, axis[:, 2]) + (axis[:, 2] - old_ground)
    pts = np.column_stack([xy, top])
    snapped = _snap(ctx, pts, p.road_side, p.width, p.gutter_width)
    update: dict[str, Any] = {}
    if snapped is not None:
        update["axis"] = [(float(x), float(y), float(z)) for (x, y), z in zip(xy, snapped["top"])]
        if p.foot_z is not None:
            update["foot_z"] = [float(v) for v in snapped["foot"]]
            update["road_side"] = int(snapped["side"])
        update.update(_gutter_update(snapped, float(p.gutter_width) if snapped.get("gutter") else 0.0))
    else:
        update["axis"] = [(float(x), float(y), float(z)) for (x, y), z in zip(xy, top)]
        if p.foot_z is not None:
            update["foot_z"] = [float(f + (t - z0)) for f, t, z0 in zip(p.foot_z, top, axis[:, 2])]
        update.update({"gutter_axis": None, "gutter_z_outer": None})
    evidence = dict(curb.evidence)
    evidence["length_m"] = _r(polyline_length(xy))
    return _replaced(curb, update, "axis", evidence=evidence)


def _op_set_curb(ctx: _Ctx, op: EditOp) -> list[Change]:
    curb = ctx.get(op.target, "curb")
    p, params = curb.params, op.params
    axis = np.asarray(p.axis, dtype=np.float64)
    has_foot = p.foot_z is not None and len(p.foot_z) == len(axis)
    current_height = element_height(curb) or float(p.height)
    height = _value(params, "height", current_height, *CURB_HEIGHT_RANGE_M)
    width = _value(params, "width", p.width, *CURB_WIDTH_RANGE_M)
    gutter = _value(params, "gutter_width", p.gutter_width, *GUTTER_WIDTH_RANGE_M)
    if height is None and width is None and gutter is None:
        raise EditError("set_curb needs height, width or gutter_width.", key=BAD_PARAMS_KEY, params={"param": "height"})
    update: dict[str, Any] = {}
    names: list[str] = []
    if height is not None:
        if has_foot:
            foot = np.asarray(p.foot_z, dtype=np.float64)
            update["axis"] = [(float(x), float(y), float(f + height)) for (x, y, _z), f in zip(axis, foot)]
            names.append("axis")
        else:
            update["height"] = height
            names.append("height")
    new_width = float(width if width is not None else p.width)
    new_gutter = float(gutter if gutter is not None else p.gutter_width)
    if width is not None:
        update["width"] = new_width
        names.append("width")
    if gutter is not None or (width is not None and new_gutter > 0):
        if new_gutter > 0 and (not p.road_side or not has_foot):
            raise EditError(f"Curb {curb.id} has no known road side; a gutter cannot be placed.", key=NO_ROAD_SIDE_KEY, params={"id": curb.id})
        pts = np.asarray(update.get("axis", p.axis), dtype=np.float64)
        snapped = _snap(ctx, pts, p.road_side, new_width, new_gutter) if new_gutter > 0 else None
        update.update(_gutter_update(snapped, new_gutter))
        if gutter is not None:
            names.append("gutter_width")
    evidence = dict(curb.evidence)
    if gutter is not None:
        evidence["gutter_width_m"] = _r(new_gutter)
        evidence["gutter"] = "manual" if new_gutter > 0 else "none"
    changed = _replaced(curb, update, *names, evidence=evidence)
    ctx.put(changed)
    return [Change(curb.id, "curb", "modified", snapshot(ctx.scene, curb), snapshot(ctx.scene, changed))]


# ------------------------------------------------------------------------------------------------- full-tier ops

def _op_set_pavement_kind(ctx: _Ctx, op: EditOp) -> list[Change]:
    pavement = ctx.get(op.target, "pavement")
    kind = str(op.params.get("kind") or "")
    if kind not in ("road", "sidewalk", "path"):
        raise EditError("kind must be road, sidewalk or path.", key=BAD_PARAMS_KEY, params={"param": "kind"})
    if kind == pavement.params.kind:
        raise EditError(f"{pavement.id} is already a {kind}.", key=INVALID_KEY, params={"param": "kind"})
    changed = _replaced(pavement, {"kind": kind}, "kind")
    ctx.put(changed)
    return [Change(pavement.id, "pavement", "modified", snapshot(ctx.scene, pavement), snapshot(ctx.scene, changed))]


def _op_delete_building(ctx: _Ctx, op: EditOp) -> list[Change]:
    building = ctx.get(op.target, "building")
    changes = [Change(building.id, "building", "removed", snapshot(ctx.scene, building), None)]
    for element in list(ctx.scene.elements.values()):
        if element.id != building.id and getattr(element, "building_id", None) == building.id:
            changes.append(Change(element.id, kind_of(element), "removed", snapshot(ctx.scene, element), None))
            ctx.drop(element.id)
    ctx.drop(building.id)
    ctx.scene.buildings.pop(building.id, None)
    return changes


def _op_set_building(ctx: _Ctx, op: EditOp) -> list[Change]:
    building = ctx.get(op.target, "building")
    heights = building_heights(building)
    ground, eave, ridge = heights["ground"], heights["eave"], heights["ridge"]
    new_eave = _value(op.params, "eave_height", eave - ground, *BUILDING_EAVE_RANGE_M)
    new_ridge = _value(op.params, "ridge_height", ridge - ground, BUILDING_EAVE_RANGE_M[0], BUILDING_RIDGE_MAX_M)
    if new_eave is None and new_ridge is None:
        raise EditError("set_building needs eave_height or ridge_height.", key=BAD_PARAMS_KEY, params={"param": "eave_height"})
    flat = ridge - eave < 1e-3
    if flat:  # flat roof: eave and ridge are one height
        target_eave = target_ridge = ground + (new_eave if new_eave is not None else float(new_ridge))
    else:
        target_eave = ground + new_eave if new_eave is not None else eave
        target_ridge = ground + new_ridge if new_ridge is not None else ridge + (target_eave - eave)
    if target_ridge < target_eave - 1e-6:
        raise EditError("The ridge must not lie below the eave.", key=INVALID_KEY, params={"param": "ridge_height"})
    vertices = np.asarray(building.params.vertices, dtype=np.float64).copy()
    upper = vertices[:, 2] >= eave - 1e-6
    if flat:
        vertices[upper, 2] += target_eave - eave
    else:
        scale = (target_ridge - target_eave) / (ridge - eave)
        vertices[upper, 2] = target_eave + (vertices[upper, 2] - eave) * scale
    changed = _replaced(building, {"vertices": [tuple(map(float, v)) for v in vertices]}, "vertices")
    ctx.put(changed)
    ctx.notes.append(f"{building.id}: roof heights changed; facade walls keep their measured height")
    return [Change(building.id, "building", "modified", snapshot(ctx.scene, building), snapshot(ctx.scene, changed))]


def _polygon_arg(ctx: _Ctx, value: Any) -> Any:
    from shapely.geometry import Polygon

    try:
        ring = [ctx.local_xy(float(point[0]), float(point[1])) for point in value]
    except (TypeError, ValueError, IndexError):
        raise EditError("polygon must be a list of [e, n] LV95 points.", key=BAD_PARAMS_KEY, params={"param": "polygon"}) from None
    if len(ring) < 3:
        raise EditError("polygon needs at least 3 points.", key=BAD_PARAMS_KEY, params={"param": "polygon"})
    polygon = Polygon(ring)
    if not polygon.is_valid:
        polygon = polygon.buffer(0)
    if polygon.is_empty or polygon.area < 0.01 or polygon.area > 20000.0:
        raise EditError("polygon area must be 0.01-20000 m2.", key=BAD_PARAMS_KEY, params={"param": "polygon"})
    return polygon


def _op_adjust_terrain(ctx: _Ctx, op: EditOp) -> list[Change]:
    import shapely

    terrains = elements_of(ctx.scene, "terrain")
    if not terrains:
        raise EditError("The outdoor model has no terrain.", key=NO_TERRAIN_KEY)
    terrain = terrains[0]
    params = op.params
    polygon = _polygon_arg(ctx, params.get("polygon"))
    dz, level = _number(params, "dz_m"), _number(params, "level_z")
    if (dz is None) == (level is None):
        raise EditError("Give either dz_m or level_z.", key=BAD_PARAMS_KEY, params={"param": "dz_m"})
    taper = _in_range("taper_m", _number(params, "taper_m") if params.get("taper_m") is not None else TERRAIN_DEFAULT_TAPER_M, *TERRAIN_TAPER_RANGE_M)
    vertices = np.asarray(terrain.params.vertices, dtype=np.float64).copy()
    points = shapely.points(vertices[:, :2])
    inside = np.asarray(shapely.contains(polygon, points), dtype=bool)
    if not inside.any():
        raise EditError("No terrain vertex lies inside the polygon.", key=NO_MATCH_KEY, params={"kind": "terrain"})
    # vertices shared with a pavement edge keep their height: the pavement stays flush with the terrain
    paved = set()
    for pavement in elements_of(ctx.scene, "pavement"):
        paved.update((round(float(x), 4), round(float(y), 4)) for x, y, _z in pavement.params.vertices)
    fixed = np.array([(round(float(x), 4), round(float(y), 4)) in paved for x, y in vertices[:, :2]], dtype=bool) if paved else np.zeros(len(vertices), bool)
    distance = np.asarray(shapely.distance(polygon.exterior, points), dtype=np.float64)
    weight = np.ones(len(vertices)) if taper <= 0 else np.clip(distance / taper, 0.0, 1.0)
    weight = np.where(inside & ~fixed, weight, 0.0)
    if level is not None:
        z_local = float(level) - float(ctx.offset[2])
        current = vertices[inside, 2]
        if abs(z_local - float(np.median(current))) > 10.0:
            raise EditError("level_z lies more than 10 m from the terrain there.", key=INVALID_KEY, params={"param": "level_z"})
        delta = weight * (z_local - vertices[:, 2])
    else:
        _in_range("dz_m", abs(float(dz)), 0.0, TERRAIN_DZ_MAX_M)
        delta = weight * float(dz)
    if not (np.abs(delta) > 1e-6).any():
        raise EditError("The terrain does not change inside the polygon.", key=NO_MATCH_KEY, params={"kind": "terrain"})
    vertices[:, 2] += delta
    changed = _replaced(terrain, {"vertices": [tuple(map(float, v)) for v in vertices]}, "vertices")
    ctx.put(changed)
    changes = [Change(terrain.id, "terrain", "modified", snapshot(ctx.scene, terrain), snapshot(ctx.scene, changed),
                      f"{int((np.abs(delta) > 1e-6).sum())} terrain vertices changed")]
    # dependents inside the (tapered) area follow the new terrain surface
    area = polygon.buffer(0.05)
    for element in list(ctx.scene.elements.values()):
        kind = kind_of(element)
        anchor = element_anchor(element)
        if kind == "tree" and anchor is not None and area.contains(shapely.points(anchor)):
            moved = _tree_at(ctx, element, *anchor)
            moved = moved.model_copy(update={"param_origin": element.param_origin})
        elif kind == "manhole" and anchor is not None and area.contains(shapely.points(anchor)):
            index = int(np.argmin(np.hypot(vertices[:, 0] - anchor[0], vertices[:, 1] - anchor[1])))
            shift = float(delta[index])
            p = element.params
            moved = element.model_copy(update={"params": p.model_copy(update={"center": (float(p.center[0]), float(p.center[1]), float(p.center[2]) + shift)})})
        elif kind in ("fence", "site_wall") and element.params.axis is not None and shapely.intersects(area, shapely.linestrings(_xy(element.params.axis))):
            moved = _with_axis(ctx, kind, element, _xy(element.params.axis))
            moved = moved.model_copy(update={"param_origin": element.param_origin})
        else:
            continue
        if moved.model_dump(mode="json") != element.model_dump(mode="json"):
            ctx.put(moved)
            changes.append(Change(element.id, kind, "modified", snapshot(ctx.scene, element), snapshot(ctx.scene, moved), "follows the terrain"))
    return changes


# ------------------------------------------------------------------------------------------------------- op table

_TARGET = {"type": "string", "description": "Element id or its short label ref, e.g. B9 (list_elements / the selection)."}
_POINT = {"type": "object", "description": "LV95 position {e, n} in metres.",
          "properties": {"e": {"type": "number"}, "n": {"type": "number"}}, "required": ["e", "n"]}
_BOX = {"type": "object", "description": "LV95 box {e: [e0, e1], n: [n0, n1]} (at most 60 m).",
        "properties": {"e": {"type": "array", "items": {"type": "number"}, "minItems": 2, "maxItems": 2},
                       "n": {"type": "array", "items": {"type": "number"}, "minItems": 2, "maxItems": 2}}, "required": ["e", "n"]}
_MANUAL = {"type": "boolean", "description": "Only after the user explicitly confirmed: add without point-cloud evidence."}


def _n(description: str, **extra: Any) -> dict[str, Any]:
    return {"type": "number", "description": description, **extra}


def _move_props() -> dict[str, Any]:
    return {"dx_m": _n("Shift east in metres."), "dy_m": _n("Shift north in metres."),
            "to": {**_POINT, "description": "Or the new LV95 position {e, n}."}}


def _with_delta(props: dict[str, Any]) -> dict[str, Any]:
    out = dict(props)
    for name, schema in props.items():
        if schema.get("type") == "number":
            out[f"{name}_delta"] = _n(f"Change of {name} in metres (instead of {name}).")
    return out


@dataclass(frozen=True)
class OutdoorOpSpec:
    name: str
    target_kind: str | None
    tier: str
    handler: Callable[[_Ctx, EditOp], list[Change]] | None
    description: str
    properties: dict[str, Any] = field(default_factory=dict)
    required: tuple[str, ...] = ()
    expensive: bool = False

    @property
    def schema(self) -> dict[str, Any]:
        properties = ({"target": dict(_TARGET)} if self.target_kind else {}) | copy.deepcopy(self.properties)
        required = (["target"] if self.target_kind else []) + list(self.required)
        return {"type": "object", "properties": properties, "required": required, "additionalProperties": False}


def _spec(name: str, kind: str | None, tier: str, handler: Any, description: str, properties: dict[str, Any] | None = None,
          required: tuple[str, ...] = (), expensive: bool = False) -> OutdoorOpSpec:
    return OutdoorOpSpec(name, kind, tier, handler, description, properties or {}, required, expensive)


_LINE_SET = {"fence": ("height", "thickness"), "site_wall": ("height", "thickness")}
OPS: dict[str, OutdoorOpSpec] = {spec.name: spec for spec in [
    _spec("add_tree", None, TIER_ELEMENTS, _op_add_tree,
          "Add a tree measured from the point cloud (canopy / trunk points around point within radius_m, default 3 m, or inside box): "
          "position, height, crown diameter / shape and trunk diameter are measured. Without evidence it fails with no_evidence; "
          "manual=true (only after the user confirmed) places a stylised tree with the given or default dimensions.",
          {"point": _POINT, "box": _BOX, "radius_m": _n("Search radius around point (0.2-15 m)."), "manual": _MANUAL,
           "height": _n("Manual only: height in metres."), "crown_diameter": _n("Manual only: crown diameter in metres."),
           "trunk_diameter": _n("Manual only: trunk diameter in metres."),
           "crown_shape": {"type": "string", "enum": list(CROWN_SHAPES), "description": "Manual only."}}),
    _spec("move_tree", "tree", TIER_ELEMENTS, _op_move_tree, "Move a tree (its base follows the terrain).", _move_props()),
    _spec("delete_tree", "tree", TIER_ELEMENTS, _op_delete("tree"), "Delete a tree. Needs the user's confirmation."),
    _spec("set_tree", "tree", TIER_ELEMENTS, _op_set_tree,
          "Change tree dimensions in metres: height 1-40, crown_diameter 0.5-25, trunk_diameter 0.05-3 (or *_delta), crown_shape.",
          _with_delta({"height": _n("Tree height in metres."), "crown_diameter": _n("Crown diameter in metres."),
                       "trunk_diameter": _n("Trunk diameter in metres.")})
          | {"crown_shape": {"type": "string", "enum": list(CROWN_SHAPES)}}),
    _spec("add_manhole", None, TIER_ELEMENTS, _op_add_manhole,
          "Add a manhole measured from the manhole-cover points (around point within radius_m, default 1 m, or inside box): "
          "centre, cover height and diameter are measured. Without evidence it fails with no_evidence; manual=true only after "
          "the user confirmed (cover at the terrain, diameter as given or 0.6 m).",
          {"point": _POINT, "box": _BOX, "radius_m": _n("Search radius around point (0.2-15 m)."), "manual": _MANUAL,
           "diameter": _n("Manual only: cover diameter in metres."), "type": {"type": "string", "description": "Cover type label."}}),
    _spec("move_manhole", "manhole", TIER_ELEMENTS, _op_move_manhole, "Move a manhole (cover height follows the terrain).", _move_props()),
    _spec("delete_manhole", "manhole", TIER_ELEMENTS, _op_delete("manhole"), "Delete a manhole. Needs the user's confirmation."),
    _spec("set_manhole", "manhole", TIER_ELEMENTS, _op_set_manhole,
          "Change a manhole: diameter 0.3-1.5 m (or diameter_delta), cover_z = absolute LN02 cover height (or cover_z_delta), type.",
          {"diameter": _n("Cover diameter in metres."), "diameter_delta": _n("Change of the diameter in metres."),
           "cover_z": _n("Absolute cover height LN02 in metres."), "cover_z_delta": _n("Change of the cover height in metres."),
           "type": {"type": "string"}}),
    _spec("delete_fence", "fence", TIER_ELEMENTS, _op_delete("fence"), "Delete a fence. Needs the user's confirmation."),
    _spec("set_fence", "fence", TIER_ELEMENTS, _op_set_fence,
          "Change a fence: height 0.3-3 m above the terrain, thickness 0.01-0.5 m (or *_delta).",
          _with_delta({"height": _n("Height above the terrain in metres."), "thickness": _n("Thickness in metres.")})),
    _spec("offset_fence", "fence", TIER_ELEMENTS, _op_offset_line("fence"),
          "Move a fence parallel by distance_m (positive = to the left looking from its first to its last vertex; max 5 m).",
          {"distance_m": _n("Parallel offset in metres.")}, ("distance_m",)),
    _spec("move_fence_vertex", "fence", TIER_ELEMENTS, _op_move_vertex("fence"),
          "Move one fence vertex (index from element_details, 0 = first, -1 = last) by dx_m east / dy_m north.",
          {"index": {"type": "integer"}, "dx_m": _n("Shift east in metres."), "dy_m": _n("Shift north in metres.")}, ("index",)),
    _spec("delete_site_wall", "site_wall", TIER_ELEMENTS, _op_delete("site_wall"), "Delete a garden / site wall. Needs the user's confirmation."),
    _spec("set_site_wall", "site_wall", TIER_ELEMENTS, _op_set_site_wall,
          "Change a garden / site wall: height 0.2-4 m above its base, thickness 0.05-1.5 m (or *_delta).",
          _with_delta({"height": _n("Height above the base in metres."), "thickness": _n("Thickness in metres.")})),
    _spec("offset_site_wall", "site_wall", TIER_ELEMENTS, _op_offset_line("site_wall"),
          "Move a site wall parallel by distance_m (positive = to the left looking from its first to its last vertex; max 5 m).",
          {"distance_m": _n("Parallel offset in metres.")}, ("distance_m",)),
    _spec("move_site_wall_vertex", "site_wall", TIER_ELEMENTS, _op_move_vertex("site_wall"),
          "Move one site-wall axis vertex (index from element_details) by dx_m east / dy_m north.",
          {"index": {"type": "integer"}, "dx_m": _n("Shift east in metres."), "dy_m": _n("Shift north in metres.")}, ("index",)),
    _spec("delete_curb", "curb", TIER_ELEMENTS, _op_delete("curb"), "Delete a curb. Needs the user's confirmation."),
    _spec("set_curb", "curb", TIER_ELEMENTS, _op_set_curb,
          "Change a curb: height = reveal above the road side 0.01-0.3 m, width 0.05-0.5 m, gutter_width 0-1 m (0 = no gutter), or *_delta.",
          _with_delta({"height": _n("Reveal in metres."), "width": _n("Curb width in metres."), "gutter_width": _n("Gutter width in metres.")})),
    _spec("offset_curb", "curb", TIER_ELEMENTS, _op_offset_line("curb"),
          "Move a curb parallel by distance_m (positive = to the left of its direction; max 5 m); it is snapped to the terrain again.",
          {"distance_m": _n("Parallel offset in metres.")}, ("distance_m",)),
    # ------------------------------------------------------------------------------------------- full control only
    _spec("delete_pavement", "pavement", TIER_FULL, _op_delete("pavement"), "Delete a pavement surface (road / sidewalk / path)."),
    _spec("set_pavement_kind", "pavement", TIER_FULL, _op_set_pavement_kind, "Change the kind of a pavement surface.",
          {"kind": {"type": "string", "enum": ["road", "sidewalk", "path"]}}, ("kind",)),
    _spec("delete_building", "building", TIER_FULL, _op_delete_building, "Delete a building with its facade elements."),
    _spec("set_building", "building", TIER_FULL, _op_set_building,
          "Change the eave and / or ridge height of a building in metres above its lowest point (the roof part is scaled vertically).",
          _with_delta({"eave_height": _n("Eave height above ground in metres."), "ridge_height": _n("Ridge height above ground in metres.")})),
    _spec("adjust_terrain", None, TIER_FULL, _op_adjust_terrain,
          "Raise / lower the terrain inside an LV95 polygon by dz_m, or level it to level_z (LN02); taper_m (default 0.5) keeps the "
          "edge flush, pavement edges stay fixed; trees, manholes, fences and walls inside follow. May take longer.",
          {"polygon": {"type": "array", "items": {"type": "array", "items": {"type": "number"}, "minItems": 2, "maxItems": 2},
                       "minItems": 3, "description": "Closed ring of [e, n] LV95 points."},
           "dz_m": _n("Height change in metres (max 5)."), "level_z": _n("Absolute LN02 height to level to."),
           "taper_m": _n("Transition width at the polygon edge in metres (0-5).")}, ("polygon",), expensive=True),
    _spec(BULK_OP, None, TIER_FULL, None,
          "Apply one edit operation to every element of a category matching filter (all or nothing, max 500), e.g. all trees "
          "0.5 m higher: category tree, op set_tree, params {height_delta: 0.5}. May take longer.",
          {"category": {"type": "string", "enum": [*ELEMENT_KINDS, "pavement", "building"]},
           "filter": {"type": "object", "description": "Optional: ids (list), box {e, n} (LV95), min_height / max_height, kind "
                                                       "(pavement kind), crown_shape, label (text in id or name).",
                      "properties": {"ids": {"type": "array", "items": {"type": "string"}}, "box": _BOX,
                                     "min_height": {"type": "number"}, "max_height": {"type": "number"},
                                     "kind": {"type": "string"}, "crown_shape": {"type": "string"}, "label": {"type": "string"}}},
           "op": {"type": "string", "description": "Name of a single-element edit tool of that category (set_*, delete_*, offset_*, move_*)."},
           "params": {"type": "object", "description": "Parameters of that operation (without target)."}},
          ("category", "op"), expensive=True),
]}


def ops_for_tier(tier: str) -> list[OutdoorOpSpec]:
    """The edit operations of a tool tier (``"full"`` includes the elements tier)."""
    return [spec for spec in OPS.values() if spec.tier == TIER_ELEMENTS or tier == TIER_FULL]


def _check_params(spec: OutdoorOpSpec, op: EditOp) -> None:
    allowed = set(spec.properties)
    unknown = [key for key in op.params if key not in allowed and not str(key).startswith("_")]
    if unknown:
        raise EditError(f"Unknown parameter(s) {', '.join(sorted(unknown))} for {op.op}.", key=BAD_PARAMS_KEY, params={"param": unknown[0]})
    if spec.target_kind is not None and not op.target:
        raise EditError(f"{op.op} needs a target.", key=BAD_PARAMS_KEY, params={"param": "target"})
    for key in spec.required:
        if op.params.get(key) is None:
            raise EditError(f"Parameter {key!r} is missing.", key=BAD_PARAMS_KEY, params={"param": key})


def select_ids(scene: Any, category: str, filter_spec: dict[str, Any] | None) -> list[str]:
    """Ids of the elements of ``category`` matching a bulk filter (see ``apply_to_category``)."""
    spec = dict(filter_spec or {})
    ids = spec.get("ids")
    wanted = {str(i) for i in ids} if isinstance(ids, list) and ids else None
    box = spec.get("box")
    bounds = None
    if box is not None:
        try:
            offset = origin_offset(scene)
            e0, e1 = sorted(float(v) for v in box["e"])
            n0, n1 = sorted(float(v) for v in box["n"])
            bounds = (e0 - offset[0], e1 - offset[0], n0 - offset[1], n1 - offset[1])
        except (KeyError, TypeError, ValueError):
            raise EditError("filter.box must be {e: [e0, e1], n: [n0, n1]}.", key=BAD_PARAMS_KEY, params={"param": "filter"}) from None
    low, high = _number(spec, "min_height"), _number(spec, "max_height")
    label = str(spec.get("label") or "").strip().lower()
    out: list[str] = []
    for element in elements_of(scene, category):
        if wanted is not None and element.id not in wanted:
            continue
        if label and label not in element.id.lower() and label not in str(element.name or "").lower():
            continue
        if spec.get("kind") and category == "pavement" and element.params.kind != spec["kind"]:
            continue
        if spec.get("crown_shape") and category == "tree" and (element.params.crown_shape or "ellipsoid") != spec["crown_shape"]:
            continue
        if low is not None or high is not None:
            height = element_height(element)
            if height is None or (low is not None and height < low) or (high is not None and height > high):
                continue
        if bounds is not None:
            line = _line_points(element)
            points = line if line is not None else ([element_anchor(element)] if element_anchor(element) is not None else [])
            if not any(bounds[0] <= x <= bounds[1] and bounds[2] <= y <= bounds[3] for x, y in points):
                continue
        out.append(element.id)
    return out


# --------------------------------------------------------------------------------------------------------- proposal

def _label_changes(changes: list[Change], before: dict[str, str], after: dict[str, str]) -> None:
    """Put the short display label (``ref``) into the before / after snapshots of the changes (summary, change list)."""
    for change in changes:
        old, new = before.get(change.element_id), after.get(change.element_id)
        if isinstance(change.before, dict) and (old or new):
            change.before = {**change.before, "ref": old or new}
        if isinstance(change.after, dict) and (new or old):
            change.after = {**change.after, "ref": new or old}


class OutdoorProposal:
    """Edits applied on a copy-on-write copy of ``base`` (the base Scene and its elements are never mutated)."""

    def __init__(self, base: Any, *, support: EnvSupport | None = None, project: Any = None) -> None:
        self.base = base
        self.plan = base  # the edited Scene (name shared with the indoor proposal)
        self.ops: list[EditOp] = []
        self.changes: list[Change] = []
        self.notes: list[str] = []
        self.signatures: dict[str, dict[str, Any]] = {}
        self.created: list[str] = []
        self.confirm_reasons: list[str] = []
        self._support = support
        self._project = project

    @property
    def scene(self) -> Any:
        return self.plan

    @property
    def support(self) -> EnvSupport:
        if self._support is None:
            self._support = EnvSupport.for_project(self._project)
        return self._support

    def apply(self, op: EditOp | dict[str, Any], *, cloud: Any | None = None, replay: bool = False) -> list[Change]:
        """Apply one op (all or nothing); raises :class:`EditError` and leaves the proposal unchanged on failure.

        Parameters starting with ``_`` are internal (the element of an accepted add): they are dropped from incoming
        ops and only honoured with ``replay=True``.  ``apply_to_category`` is expanded into one op per element.
        """
        op = EditOp.from_dict(op) if isinstance(op, dict) else EditOp(op.op, op.target, copy.deepcopy(op.params))
        if not replay:
            op.params = {key: value for key, value in op.params.items() if not str(key).startswith("_")}
        if op.op == BULK_OP:
            return self._apply_bulk(op, cloud=cloud)
        spec = OPS.get(op.op)
        if spec is None or spec.handler is None:
            raise EditError(f"Unknown edit operation {op.op!r}.", key="assistant.edit.unknown_op", params={"op": op.op})
        _check_params(spec, op)
        from samcloud.assistant.domains.outdoor_labels import display_labels, resolve_label

        if op.target and op.target not in self.plan.elements and op.target not in self.created:
            real = resolve_label(self.plan, op.target)  # a short label ("B9") -> the real id, stored in the history
            if real is not None:
                op.target = real
        labels_before = display_labels(self.plan)
        work = _work_copy(self.plan)
        ctx = _Ctx(work, self.support, cloud, replay)
        signature = None
        if op.target and op.target not in self.created:
            signature = signature_of(self.base, op.target) or signature_of(self.plan, op.target)
        changes = spec.handler(ctx, op)
        _label_changes(changes, labels_before, display_labels(work))
        created = [change.element_id for change in changes if change.change == "added"]
        if created:
            ctx.resolved["_created_ids"] = created
        op.params.update(ctx.resolved)
        if op.target and signature is not None and op.target not in self.signatures:
            self.signatures[op.target] = signature
        self.created.extend(created)
        self.plan = work
        self.ops.append(op)
        self.changes.extend(changes)
        self.notes.extend(ctx.notes)
        self.confirm_reasons.extend(ctx.confirm)
        return changes

    def _apply_bulk(self, op: EditOp, *, cloud: Any | None) -> list[Change]:
        params = op.params
        category, name = str(params.get("category") or ""), str(params.get("op") or "")
        spec = OPS.get(name)
        if spec is None or spec.handler is None or spec.target_kind is None:
            raise EditError(f"{name!r} is not a single-element edit operation.", key="assistant.edit.unknown_op", params={"op": name})
        if spec.target_kind != category:
            raise EditError(f"{name} edits {spec.target_kind}, not {category}.", key=BAD_PARAMS_KEY, params={"param": "op"})
        sub_params = params.get("params") or {}
        if not isinstance(sub_params, dict):
            raise EditError("params must be an object.", key=BAD_PARAMS_KEY, params={"param": "params"})
        unknown = [key for key in params if key not in ("category", "op", "params", "filter")]
        if unknown:
            raise EditError(f"Unknown parameter(s) {', '.join(sorted(unknown))} for {BULK_OP}.", key=BAD_PARAMS_KEY, params={"param": unknown[0]})
        ids = select_ids(self.plan, category, params.get("filter") if isinstance(params.get("filter"), dict) else None)
        if not ids:
            raise EditError(f"No {category} matches the filter.", key=NO_MATCH_KEY, params={"kind": category})
        if len(ids) > MAX_BULK_ELEMENTS:
            raise EditError(f"{len(ids)} elements match; at most {MAX_BULK_ELEMENTS} at once.", key=INVALID_KEY,
                            params={"param": "filter", "max": MAX_BULK_ELEMENTS})
        state = (self.plan, list(self.ops), list(self.changes), list(self.notes), dict(self.signatures), list(self.created),
                 list(self.confirm_reasons))
        changes: list[Change] = []
        current = None
        try:
            for element_id in ids:
                current = element_id
                changes.extend(self.apply(EditOp(name, element_id, copy.deepcopy(sub_params)), cloud=cloud))
        except EditError as error:
            (self.plan, self.ops, self.changes, self.notes, self.signatures, self.created, self.confirm_reasons) = state
            raise EditError(f"{name} on {current}: {error.message}", key=error.key, params={**error.params, "id": current}) from None
        self.notes.append(f"{name} applied to {len(ids)} {category} element(s)")
        return changes

    def summary(self) -> list[dict[str, Any]]:
        """Before -> after per changed element (first before, last after) for the UI."""
        order: list[str] = []
        merged: dict[str, dict[str, Any]] = {}
        for change in self.changes:
            entry = merged.get(change.element_id)
            if entry is None:
                order.append(change.element_id)
                merged[change.element_id] = {"element_id": change.element_id, "kind": change.kind, "change": change.change,
                                             "before": copy.deepcopy(change.before), "after": copy.deepcopy(change.after),
                                             "notes": [change.note] if change.note else []}
                ref = (change.after or {}).get("ref") or (change.before or {}).get("ref")
                if ref:
                    merged[change.element_id]["ref"] = ref
                continue
            entry["after"] = copy.deepcopy(change.after)
            if (change.after or {}).get("ref"):
                entry["ref"] = change.after["ref"]
            if change.note and change.note not in entry["notes"]:
                entry["notes"].append(change.note)
            if entry["change"] == "added":
                entry["change"] = "added" if change.change != "removed" else "discarded"
            elif change.change == "removed":
                entry["change"] = "removed"
            elif entry["change"] != change.change:
                entry["change"] = "modified"
        return [merged[key] for key in order if merged[key]["change"] != "discarded"]

    def changed_ids(self) -> set[str]:
        return {entry["element_id"] for entry in self.summary()}

    def needs_confirmation(self) -> bool:
        """Any removal or more than :data:`CONFIRM_ELEMENT_COUNT` changed elements."""
        entries = self.summary()
        return bool(self.confirm_reasons) or any(entry["change"] == "removed" for entry in entries) or len(entries) > CONFIRM_ELEMENT_COUNT


def apply_ops(base: Any, ops: list[EditOp | dict[str, Any]], *, cloud: Any | None = None, support: EnvSupport | None = None,
              project: Any = None) -> OutdoorProposal:
    """A proposal with ``ops`` applied in order (raises on the first failing op)."""
    proposal = OutdoorProposal(base, support=support, project=project)
    for op in ops:
        proposal.apply(op, cloud=cloud)
    return proposal


__all__ = ["BULK_OP", "ELEMENT_KINDS", "FULL_KINDS", "MANUAL_EVIDENCE", "OPS", "OutdoorOpSpec", "OutdoorProposal", "apply_ops",
           "building_heights", "diff_scenes", "element_anchor", "element_height", "elements_of", "find_element", "is_manual",
           "kind_of", "mesh_footprint", "offset_polyline", "ops_for_tier", "polyline_length", "resolve_signature", "select_ids",
           "signature_of", "snapshot", "surface_heights"]
