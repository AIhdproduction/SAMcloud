"""Build a SAMcloud :class:`Scene` from a persisted ``BuildingPlan``.

Everything is in metres in the local plan frame (origin offset 0).  The plan frame (origin,
scale, rotation, units per metre) is kept in ``scene.derived["samcloud_frame"]`` and written to the
IFC project as property set ``SAMcloud_Frame``, so georeferencing can be added later.

Fixes for the problems seen when the old Raumlot pipeline ran its own wall extraction on the same flat:

* duplicate / stacked wall pieces are merged (collinear overlapping or touching pieces),
* wall ends are joined (L / T joins, the core cuts the footprints so ends meet exactly),
* spaces come from the plan rooms (``SpaceParams.outline``), not from the wall network,
* ceilings are per room and clipped by the wall footprints (they end at the inner wall faces),
* openings that sit in a gap between two wall pieces bridge the gap (the wall becomes one
  continuous host); openings without any host are skipped with a warning,
* an opening has both faces of its host: merging pieces never moves them, a measured step between two pieces stays, the
  door reveal belongs to the rooms and no ceiling runs into the lintel above it (``samcloud.bim.opening_flush`` is the check).
"""

from __future__ import annotations

import math
import uuid
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from shapely import make_valid
from shapely.geometry import LineString, MultiPolygon, Point, Polygon
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union

from samcloud.bim.core.geometry.walls import compute_wall_geometries
from samcloud.bim.core.model import (
    CeilingElement,
    CeilingIfcClass,
    DataSource,
    DoorElement,
    DoorOperation,
    FurnitureElement,
    IndoorDetail,
    JoinEnd,
    JoinType,
    OpeningElement,
    ParamOrigin,
    ProjectSettings,
    Scene,
    Side,
    SlabElement,
    SpaceElement,
    WallElement,
    WallFunction,
    WallJoinElement,
    WindowElement,
)
from samcloud.bim.core.model.elements import (
    CeilingParams,
    DoorParams,
    OpaqueParams,
    OpeningParams,
    PropertyValue,
    SlabParams,
    SpaceParams,
    WallJoinParams,
    WallParams,
    WindowParams,
)
from samcloud.bim.core.model.scene import Building, Site, Storey
from samcloud.bim.furniture import FurnitureMapping, load_furniture_mapping
from samcloud.bim.ceiling_scene import add_ceiling_objects
from samcloud.bim.plan_model import BuildingPlan, PlanFurniture, PlanOpening, PlanWall
from samcloud.bim.structure_scene import add_structure_elements, room_ceiling_pieces, space_height_properties

NAMESPACE = uuid.UUID("3f6c1c0e-8d0a-5b0e-9c55-53414d636c64")
"""Fixed namespace of the deterministic scene element ids."""

_MIN_INTERIOR_AREA_M2 = 1e-4
_MEASURED_WALL_FACES = frozenset({"face", "depth"})
_MEASURED_WALL_HEIGHTS = frozenset({"room_ceiling", "points", "ceiling_occluded"})
"""Plan wall height sources measured from the wall's own points (``adjacent_ceiling`` is inferred from the rooms)."""
"""Smallest meaningful plan opening; smaller rings are clipping artefacts."""

_DEFAULTS: dict[str, Any] = {
    "storey": {"fallback_height_m": 2.5},
    "slab": {"thickness_m": 0.20, "close_gap_m": 0.02, "simplify_m": 0.01, "min_hole_area_m2": 0.5, "min_area_m2": 0.5},
    "ceiling": {"thickness_m": 0.10},
    "walls": {
        "default_interior_thickness_m": 0.12,
        "default_exterior_thickness_m": 0.30,
        "min_length_m": 0.02,
        "dedup": {
            "max_angle_deg": 3.0,
            "offset_factor": 0.5,
            "min_overlap_share": 0.5,
            "chain_gap_m": 0.05,
            "chain_thickness_tol_m": 0.03,
        },
        "join": {"reach_m": 0.40, "min_angle_deg": 20.0},
    },
    "openings": {"host_angle_deg": 12.0, "host_offset_m": 0.12, "bridge_gap_m": 0.40, "min_width_m": 0.20, "lintel_reach_m": 1.0, "max_join_shift_m": 0.10, "flush_m": 0.005, "proven_step_m": 0.06},
    "qa": {"max_wall_overlap_share": 0.05, "max_ceiling_wall_overlap_m2": 0.0001},
    "furniture": {"min_area_m2": 1e-4, "simplify_m": 0.01},
}


def _merge(base: dict[str, Any], extra: dict[str, Any]) -> dict[str, Any]:
    result = {key: (dict(value) if isinstance(value, dict) else value) for key, value in base.items()}
    for key, value in extra.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merge(result[key], value)
        else:
            result[key] = value
    return result


def load_scene_config(overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    """Defaults + ``config/bim_scene.yaml`` + wall thicknesses of ``config/ortho_plan.yaml`` + overrides."""
    from samcloud.core.paths import CONFIG_DIRECTORY

    config = _merge(_DEFAULTS, {})
    ortho = Path(CONFIG_DIRECTORY) / "ortho_plan.yaml"
    if ortho.is_file():
        walls = (yaml.safe_load(ortho.read_text(encoding="utf-8")) or {}).get("walls") or {}
        config["walls"]["default_interior_thickness_m"] = float(
            walls.get("interior_thickness_m", config["walls"]["default_interior_thickness_m"])
        )
        config["walls"]["default_exterior_thickness_m"] = float(
            walls.get("exterior_thickness_m", config["walls"]["default_exterior_thickness_m"])
        )
    scene_yaml = Path(CONFIG_DIRECTORY) / "bim_scene.yaml"
    if scene_yaml.is_file():
        config = _merge(config, yaml.safe_load(scene_yaml.read_text(encoding="utf-8")) or {})
    return _merge(config, overrides or {})


@dataclass
class SceneBuildReport:
    """What the builder did; stored in ``scene.derived["samcloud_build"]`` and in qa.json."""

    walls_in: int = 0
    walls_out: int = 0
    duplicates_merged: int = 0
    chained: int = 0
    split_crossings: int = 0
    trimmed_overlaps: int = 0
    bridged: int = 0
    envelope_bridged: int = 0
    single_line_default_thickness: int = 0
    joins_l: int = 0
    joins_t: int = 0
    openings_in: int = 0
    openings_hosted: int = 0
    openings_skipped: int = 0
    openings_extended_wall: int = 0
    openings_lintel: int = 0
    openings_axis_overridden: int = 0
    openings_shifted: int = 0
    skipped_openings: list[dict[str, Any]] = field(default_factory=list)
    spaces: int = 0
    ceilings: int = 0
    slabs: int = 0
    furniture_in: int = 0
    furniture_out: int = 0
    wall_overlaps: list[dict[str, Any]] = field(default_factory=list)
    ceiling_wall_overlap_max_m2: float = 0.0
    room_area_plan_m2: float = 0.0
    room_area_scene_m2: float = 0.0
    room_areas: list[dict[str, Any]] = field(default_factory=list)  # per plan room: id, name, plan_m2, ifc_m2 (0 when the room was skipped)
    warnings: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------

def _uid(plan: BuildingPlan, kind: str, key: str) -> str:
    return str(uuid.uuid5(NAMESPACE, f"{plan.project_name}/{plan.level_id}/{kind}/{key}"))


def _unit(vector: np.ndarray) -> np.ndarray:
    length = float(np.linalg.norm(vector))
    return vector / length if length > 0 else vector


def _angle_between(a: np.ndarray, b: np.ndarray) -> float:
    """Acute angle between two directions (degrees, 0..90)."""
    value = abs(float(np.dot(_unit(a), _unit(b))))
    return math.degrees(math.acos(min(1.0, value)))


def _polygons(geometry: BaseGeometry) -> list[Polygon]:
    if isinstance(geometry, Polygon):
        return [] if geometry.is_empty else [geometry]
    if isinstance(geometry, MultiPolygon) or hasattr(geometry, "geoms"):
        result: list[Polygon] = []
        for part in geometry.geoms:
            result.extend(_polygons(part))
        return result
    return []


def _clean_polygon(geometry: BaseGeometry) -> Polygon | None:
    """Return the largest valid polygon, discarding non-polygonal repair remnants."""
    pieces = sorted(_polygons(make_valid(geometry)), key=lambda polygon: -polygon.area)
    return pieces[0] if pieces else None


def _interior_rings(polygon: Polygon, *, min_area_m2: float) -> list[list[tuple[float, float]]]:
    """Convert only valid, meaningful interior rings to parameter outlines."""
    rings: list[list[tuple[float, float]]] = []
    for ring in polygon.interiors:
        hole = Polygon(ring)
        if hole.is_valid and not hole.is_empty and hole.area >= min_area_m2:
            rings.append(_ring(hole, ccw=False))
    return rings


def _without_holes(polygon: Polygon) -> list[Polygon]:
    """Pieces without interior rings covering ``polygon`` (a ceiling outline is a single ring): cut through each hole."""
    if not polygon.interiors:
        return [polygon]
    middle = Polygon(polygon.interiors[0]).representative_point().x
    x0, y0, x1, y1 = polygon.bounds
    below = Polygon([(x0 - 1.0, y0 - 1.0), (middle, y0 - 1.0), (middle, y1 + 1.0), (x0 - 1.0, y1 + 1.0)])
    above = Polygon([(middle, y0 - 1.0), (x1 + 1.0, y0 - 1.0), (x1 + 1.0, y1 + 1.0), (middle, y1 + 1.0)])
    return [piece for side in (below, above) for part in _polygons(polygon.intersection(side)) for piece in _without_holes(part)]


def _ring(polygon: Polygon, *, ccw: bool = True) -> list[tuple[float, float]]:
    coords = [(float(x), float(y)) for x, y in polygon.exterior.coords[:-1]]
    area2 = sum(coords[i][0] * coords[(i + 1) % len(coords)][1] - coords[(i + 1) % len(coords)][0] * coords[i][1] for i in range(len(coords)))
    if (area2 > 0) != ccw:
        coords.reverse()
    return coords


@dataclass
class _Wall:
    """Working wall in metres: a centreline segment with thickness."""

    id: str
    p0: np.ndarray
    p1: np.ndarray
    thickness: float
    kind: str
    members: list[str]
    height: float | None = None
    single_line: bool = False
    height_source: str = "storey"  # plan wall height source (samcloud.ortho.vector.structure)
    evidence: str = ""  # plan wall evidence (how the faces were found)

    @property
    def length(self) -> float:
        return float(np.linalg.norm(self.p1 - self.p0))

    @property
    def direction(self) -> np.ndarray:
        return _unit(self.p1 - self.p0)

    def along(self, point: np.ndarray) -> float:
        return float(np.dot(point - self.p0, self.direction))

    def offset(self, point: np.ndarray) -> float:
        d = self.direction
        rel = point - self.p0
        return float(d[0] * rel[1] - d[1] * rel[0])

    def set_interval(self, lo: float, hi: float) -> None:
        origin, direction = self.p0.copy(), self.direction
        self.p0 = origin + direction * lo
        self.p1 = origin + direction * hi


# --------------------------------------------------------------------------------------
# wall network
# --------------------------------------------------------------------------------------

def _plan_walls(plan: BuildingPlan, cfg: dict[str, Any], report: SceneBuildReport) -> list[_Wall]:
    upm = plan.units_per_metre
    wcfg = cfg["walls"]
    walls: list[_Wall] = []
    for item in plan.walls:
        points = np.asarray(item.centreline, dtype=float) / upm
        if len(points) < 2:
            continue
        p0, p1 = points[0], points[-1]
        if float(np.linalg.norm(p1 - p0)) < wcfg["min_length_m"]:
            report.warnings.append(f"wall {item.id}: shorter than {wcfg['min_length_m']} m, dropped")
            continue
        thickness = float(item.thickness_m)
        single = thickness <= 0.0
        if single:
            thickness = wcfg["default_exterior_thickness_m"] if item.kind == "exterior" else wcfg["default_interior_thickness_m"]
            report.single_line_default_thickness += 1
            report.warnings.append(f"wall {item.id}: single line, default thickness {thickness:.2f} m used")
        walls.append(_Wall(item.id, p0, p1, thickness, item.kind, [item.id], item.height_m, single, str(getattr(item, "height_source", "storey") or "storey"), str(item.evidence or "")))
    return walls


def _merge_pair(keep: _Wall, other: _Wall) -> None:
    """Extend ``keep`` along its own line to the union of both intervals; keep its thickness."""
    d = keep.direction
    lo = min(0.0, float(np.dot(other.p0 - keep.p0, d)), float(np.dot(other.p1 - keep.p0, d)))
    hi = max(keep.length, float(np.dot(other.p0 - keep.p0, d)), float(np.dot(other.p1 - keep.p0, d)))
    keep.set_interval(lo, hi)
    keep.members.extend(other.members)
    measured_heights = [value for value in (keep.height, other.height) if value]
    if other.height and (not keep.height or other.height > keep.height):
        keep.height_source = other.height_source
    keep.height = max(measured_heights) if measured_heights else None
    if other.kind == "exterior" and keep.kind != "exterior":
        keep.kind = "exterior"


def _overlap_len(a: _Wall, b: _Wall) -> float:
    d = a.direction
    lo = max(0.0, min(float(np.dot(b.p0 - a.p0, d)), float(np.dot(b.p1 - a.p0, d))))
    hi = min(a.length, max(float(np.dot(b.p0 - a.p0, d)), float(np.dot(b.p1 - a.p0, d))))
    return max(0.0, hi - lo)


def _collinear(a: _Wall, b: _Wall, max_angle: float, offset_factor: float) -> bool:
    if _angle_between(a.direction, b.direction) > max_angle:
        return False
    mid = (b.p0 + b.p1) / 2.0
    return abs(a.offset(mid)) < offset_factor * max(a.thickness, b.thickness)


def _band_difference(keep: _Wall, other: _Wall, points: tuple[np.ndarray, ...]) -> float:
    """Largest distance between the faces of two parallel pieces at ``points`` (on or beside their lines)."""
    half = abs(keep.thickness - other.thickness) / 2.0
    return max(abs(keep.offset(other.p0 + other.direction * other.along(point))) for point in points) + half


def _dedup(
    walls: list[_Wall], cfg: dict[str, Any], report: SceneBuildReport, hosts: frozenset[str] = frozenset()
) -> tuple[list[_Wall], dict[str, str]]:
    """Merge duplicate and touching collinear pieces.  Pieces that only touch never chain when that moves a face of a host (``hosts``)."""
    dcfg = cfg["walls"]["dedup"]
    flush = float(cfg["openings"].get("flush_m", 0.005))
    merged_into: dict[str, str] = {}
    changed = True
    while changed:
        changed = False
        order = sorted(range(len(walls)), key=lambda i: (-walls[i].length, -walls[i].thickness, walls[i].id))
        for pos, i in enumerate(order):
            keep = walls[i]
            for j in order[pos + 1:]:
                other = walls[j]
                if other is keep or not _collinear(keep, other, dcfg["max_angle_deg"], dcfg["offset_factor"]):
                    continue
                shorter = min(keep.length, other.length)
                overlap = _overlap_len(keep, other)
                d = keep.direction
                lo = min(float(np.dot(other.p0 - keep.p0, d)), float(np.dot(other.p1 - keep.p0, d)))
                hi = max(float(np.dot(other.p0 - keep.p0, d)), float(np.dot(other.p1 - keep.p0, d)))
                gap = max(lo - keep.length, 0.0 - hi, 0.0)
                if overlap > dcfg["min_overlap_share"] * shorter:
                    report.duplicates_merged += 1
                elif gap <= dcfg["chain_gap_m"] and abs(keep.thickness - other.thickness) <= dcfg["chain_thickness_tol_m"]:
                    if hosts.intersection([*keep.members, *other.members]) and _band_difference(keep, other, (other.p0, other.p1)) > flush:
                        continue
                    report.chained += 1
                else:
                    continue
                _merge_pair(keep, other)
                merged_into[other.id] = keep.id
                walls = [w for w in walls if w is not other]
                changed = True
                break
            if changed:
                break
    # resolve chains of merges
    for key in list(merged_into):
        target = merged_into[key]
        while target in merged_into:
            target = merged_into[target]
        merged_into[key] = target
    return walls, merged_into


def _opening_spans_gap(
    first: _Wall, start: float, end: float, openings: list[PlanOpening], upm: float, cfg: dict[str, Any]
) -> bool:
    """Whether an opening occupies a proposed collinear exterior-wall bridge."""
    axis_tolerance = float(cfg["openings"]["host_angle_deg"])
    for opening in openings:
        axis = np.array([math.cos(opening.axis_angle), math.sin(opening.axis_angle)])
        if _angle_between(first.direction, axis) > axis_tolerance:
            continue
        centre = np.asarray(opening.center, dtype=float) / upm
        reach = max(first.thickness, float(opening.wall_thickness_m)) / 2.0 + float(cfg["openings"]["host_offset_m"])
        if abs(first.offset(centre)) > reach:
            continue
        half = float(opening.width_m) / 2.0
        lo = first.along(centre - axis * half)
        hi = first.along(centre + axis * half)
        # A jamb may coincide with the bridge endpoint.  Only an opening that reaches
        # materially into the bridge blocks it; a touching span is exactly the case to close.
        if max(min(lo, hi), start) + 0.005 < min(max(lo, hi), end):
            return True
    return False


def _bridge_exterior_t_gaps(
    walls: list[_Wall],
    openings: list[PlanOpening],
    upm: float,
    cfg: dict[str, Any],
    report: SceneBuildReport,
    merged_into: dict[str, str],
) -> list[_Wall]:
    """Join exterior fragments separated only by an interior T-junction.

    The IFC join kernel cuts only wall ends.  If a noisy extraction stops two collinear exterior
    runs on either side of a partition, an L join between the partition and one fragment leaves a
    full-height slot in the envelope.  The source wall bodies already prove that this is one wall;
    merge the exterior pieces before joins are made.  A real opening is always kept separate.
    """
    dcfg = cfg["walls"]["dedup"]
    reach = float(cfg["walls"]["join"]["reach_m"])
    changed = True
    while changed:
        changed = False
        exterior = sorted((wall for wall in walls if wall.kind == "exterior"), key=lambda wall: wall.id)
        for index, first in enumerate(exterior):
            for second in exterior[index + 1:]:
                if not _collinear(first, second, dcfg["max_angle_deg"], dcfg["offset_factor"]):
                    continue
                projected = sorted((first.along(second.p0), first.along(second.p1)))
                if projected[0] >= first.length:
                    start, end = first.length, projected[0]
                elif projected[1] <= 0.0:
                    start, end = projected[1], 0.0
                else:
                    continue
                gap = end - start
                # The existing join reach bounds this repair.  It is deliberately not a broad
                # facade completion: an interior endpoint must also land inside the short gap.
                if gap <= 0.005 or gap > 2.0 * reach:
                    continue
                if _opening_spans_gap(first, start, end, openings, upm, cfg):
                    continue
                has_t = False
                for candidate in walls:
                    if candidate is first or candidate is second:
                        continue
                    if _angle_between(first.direction, candidate.direction) < cfg["walls"]["join"]["min_angle_deg"]:
                        continue
                    for endpoint in (candidate.p0, candidate.p1):
                        along = first.along(endpoint)
                        contact = abs(first.offset(endpoint)) <= (first.thickness + candidate.thickness) / 2.0 + 0.01
                        if contact and start - first.thickness <= along <= end + first.thickness:
                            has_t = True
                            break
                    if has_t:
                        break
                if not has_t:
                    continue
                keep, other = (first, second) if (first.length, first.id) >= (second.length, second.id) else (second, first)
                _merge_pair(keep, other)
                merged_into[other.id] = keep.id
                walls.remove(other)
                report.envelope_bridged += 1
                changed = True
                break
            if changed:
                break
    return walls


def _trim_overlaps(
    walls: list[_Wall], cfg: dict[str, Any], report: SceneBuildReport, protected: set[int] | None = None
) -> list[_Wall]:
    """Shorten parallel pieces whose end regions overlap laterally touching bands.

    Pieces that are not collinear enough to be merged (different thickness or a small lateral
    jog) but overlap along the axis would stack their bodies.  The thinner (else shorter) piece is
    cut back to where the other starts.  A piece lying in the middle of another is left alone.
    """
    max_angle = cfg["walls"]["dedup"]["max_angle_deg"]
    changed = True
    while changed:
        changed = False
        for ia, a in enumerate(walls):
            for b in walls[ia + 1:]:
                if _angle_between(a.direction, b.direction) > max_angle:
                    continue
                if abs(a.offset((b.p0 + b.p1) / 2.0)) >= (a.thickness + b.thickness) / 2.0:
                    continue
                keep, victim = (a, b) if (a.thickness, a.length) >= (b.thickness, b.length) else (b, a)
                if protected and id(victim) in protected:
                    if id(keep) in protected:
                        continue  # both host openings: leave the overlap to the QA report
                    keep, victim = victim, keep
                d = victim.direction
                k0 = float(np.dot(keep.p0 - victim.p0, d))
                k1 = float(np.dot(keep.p1 - victim.p0, d))
                lo, hi = min(k0, k1), max(k0, k1)
                overlap = min(victim.length, hi) - max(0.0, lo)
                if overlap <= 0.005:
                    continue
                covers_start, covers_end = lo <= 0.0, hi >= victim.length
                if covers_start and covers_end:
                    walls.remove(victim)
                elif covers_start:
                    victim.set_interval(hi, victim.length)
                elif covers_end:
                    victim.set_interval(0.0, lo)
                else:
                    continue
                if victim in walls and victim.length < 0.05:
                    walls.remove(victim)
                report.trimmed_overlaps += 1
                changed = True
                break
            if changed:
                break
    return walls


def _split_crossings(walls: list[_Wall], cfg: dict[str, Any], report: SceneBuildReport) -> list[_Wall]:
    """Split walls that cross another wall (X crossing) into two walls that end at the crossing.

    The core joins only wall ends (L / T); an X crossing becomes two T joins.  The thinner wall
    (the shorter one when equal) is the one that is split.
    """
    min_angle = cfg["walls"]["join"]["min_angle_deg"]
    changed = True
    while changed:
        changed = False
        for ia, a in enumerate(walls):
            for b in walls[ia + 1:]:
                if _angle_between(a.direction, b.direction) < min_angle:
                    continue
                da, db = a.direction, b.direction
                denominator = da[0] * db[1] - da[1] * db[0]
                if abs(denominator) < 1e-9:
                    continue
                rel = b.p0 - a.p0
                t = (rel[0] * db[1] - rel[1] * db[0]) / denominator
                point = a.p0 + da * t
                sa, sb = a.along(point), b.along(point)
                margin_a, margin_b = b.thickness / 2.0 + 0.05, a.thickness / 2.0 + 0.05
                if not (margin_a < sa < a.length - margin_a and margin_b < sb < b.length - margin_b):
                    continue
                victim = a if (a.thickness, a.length) < (b.thickness, b.length) else b
                walls.remove(victim)
                for suffix, (q0, q1) in (("a", (victim.p0, point)), ("b", (point, victim.p1))):
                    walls.append(
                        _Wall(f"{victim.id}{suffix}", q0.copy(), q1.copy(), victim.thickness, victim.kind,
                              [*victim.members], victim.height, victim.single_line, evidence=victim.evidence)
                    )
                report.split_crossings += 1
                changed = True
                break
            if changed:
                break
    return walls


def _extension_blocked(host: _Wall, lo: float, hi: float, walls: list[_Wall], ignore: set[int], tolerance: float = 0.005) -> bool:
    """True when growing ``host`` to ``[lo, hi]`` (along its line) would cross another wall.

    Reaching the axis of a wall within ``tolerance`` (the flush tolerance, metres) is a corner, no crossing.
    """
    d = host.direction
    parts = []
    if lo < -tolerance:
        parts.append((lo + tolerance, 0.0))
    if hi > host.length + tolerance:
        parts.append((host.length, hi - tolerance))
    for a, b in parts:
        segment = LineString([host.p0 + d * a, host.p0 + d * b])
        for other in walls:
            if other is host or id(other) in ignore:
                continue
            if segment.crosses(LineString([other.p0, other.p1])):
                return True
    return False


def _lintel_host(
    opening: PlanOpening,
    centre: np.ndarray,
    axis: np.ndarray,
    reference: _Wall | None,
    walls: list[_Wall],
    plan: BuildingPlan,
    cfg: dict[str, Any],
    report: SceneBuildReport,
) -> tuple[_Wall, PlanOpening] | None:
    """Create a short dedicated host wall exactly over the opening width.

    Used when hosting would need a wall that crosses another wall, when no wall is near the opening, or
    when the existing pieces are too short.  The segment follows the direction of ``reference`` (the wall
    the opening sits on) or the opening axis, has the wall thickness and ends at the faces of walls it
    would otherwise cross; the opening shrinks to the free span in that case.
    """
    ocfg = cfg["openings"]
    min_angle = cfg["walls"]["join"]["min_angle_deg"]
    if reference is not None:
        d = reference.direction
        thickness = float(reference.thickness)
        c = reference.p0 + d * reference.along(centre)
        kind = reference.kind
    else:
        reach = float(ocfg.get("lintel_reach_m", 1.0))
        if not any(LineString([w.p0, w.p1]).distance(Point(centre)) <= reach for w in walls):
            return None  # nothing near enough to belong to a building: genuinely without host
        d = _unit(axis)
        thickness = float(opening.wall_thickness_m) if opening.wall_thickness_m and opening.wall_thickness_m > 0 else cfg["walls"]["default_interior_thickness_m"]
        c = centre
        kind = "interior"
    half = opening.width_m / 2.0
    lo, hi = -half, half
    segment = LineString([c + d * lo, c + d * hi])
    normal = np.array([-d[1], d[0]])
    lintel_body = LineString([c + d * lo, c + d * hi]).buffer(thickness / 2.0, cap_style=2)
    trims: list[tuple[_Wall, str, float]] = []  # (wall, end to move, new position along the wall)
    for other in walls:
        if other is reference or _angle_between(other.direction, d) < min_angle:
            continue
        body = LineString([other.p0, other.p1]).buffer(other.thickness / 2.0, cap_style=2)
        piece = segment.intersection(body)
        if piece.is_empty or piece.length <= 1e-6:
            continue
        # a wall that merely ends inside the opening span (stub / T) is cut back to the lintel face
        inside_start = lintel_body.contains(Point(other.p0))
        inside_end = lintel_body.contains(Point(other.p1))
        if inside_start != inside_end:
            end = "start" if inside_start else "end"
            far = other.p1 if inside_start else other.p0
            side = 1.0 if float(np.dot(far - c, normal)) >= 0.0 else -1.0
            od = other.direction
            denominator = float(np.dot(od, normal))
            if abs(denominator) > 1e-6:
                target = float(np.dot(c + normal * side * thickness / 2.0 - other.p0, normal)) / denominator
                if 0.05 < target < other.length and abs(target - (0.0 if inside_start else other.length)) < thickness * 4.0:
                    trims.append((other, end, target))
                    continue
        ts = [float(np.dot(np.array(pt) - c, d)) for geom in getattr(piece, "geoms", [piece]) for pt in geom.coords]
        a0, b0 = min(ts), max(ts)
        if a0 <= lo + 1e-6 and b0 >= hi - 1e-6:
            return None
        if a0 <= lo + 1e-6:
            lo = max(lo, b0)
        elif b0 >= hi - 1e-6:
            hi = min(hi, a0)
        elif a0 - lo >= hi - b0:
            hi = a0
        else:
            lo = b0
        segment = LineString([c + d * lo, c + d * hi])
    if hi - lo < ocfg["min_width_m"]:
        return None
    for other, end, target in trims:
        if end == "start":
            other.set_interval(target, other.length)
        else:
            other.set_interval(0.0, target)
        report.warnings.append(f"opening {opening.id}: wall {other.id} cut back to the lintel face")
    # The lintel reaches as high as the wall it sits in (measured wall heights; None = storey height).
    neighbours = [reference] if reference is not None else [w for w in walls if LineString([w.p0, w.p1]).distance(Point(c)) <= thickness + 0.30]
    measured = [(w.height, w.height_source) for w in neighbours if w.height]
    lintel_height, lintel_source = max(measured) if measured else (None, "storey")
    wall = _Wall(f"L_{opening.id}", c + d * lo, c + d * hi, thickness, kind, [], lintel_height, False, lintel_source)
    new_opening = opening
    if hi - lo < opening.width_m - 0.005:
        mid = c + d * ((lo + hi) / 2.0)
        new_opening = replace(
            opening, width_m=float(hi - lo), center=[float(mid[0] * plan.units_per_metre), float(mid[1] * plan.units_per_metre)]
        )
        report.warnings.append(
            f"opening {opening.id}: lintel host clipped at crossing walls, width {opening.width_m:.2f} -> {hi - lo:.2f} m"
        )
    report.openings_lintel += 1
    why = "axis inconsistent" if reference is not None and _angle_between(reference.direction, axis) > ocfg["host_angle_deg"] else "hosting would cross another wall"
    report.warnings.append(f"opening {opening.id}: dedicated lintel wall {wall.id} created ({why})")
    return wall, new_opening


def _line_reference(opening: PlanOpening, centre: np.ndarray, axis: np.ndarray, walls: list[_Wall], cfg: dict[str, Any]) -> _Wall | None:
    """The nearest piece on the line of an opening that lies in a gap (or beyond a wall end): the lintel continues its band."""
    ocfg = cfg["openings"]
    best: tuple[float, _Wall] | None = None
    for wall in walls:
        if _angle_between(wall.direction, axis) > ocfg["host_angle_deg"] or abs(wall.offset(centre)) > wall.thickness / 2.0 + ocfg["host_offset_m"]:
            continue
        along = wall.along(centre)
        gap = max(along - opening.width_m / 2.0 - wall.length, -along - opening.width_m / 2.0, 0.0)
        if gap <= ocfg["lintel_reach_m"] and (best is None or gap < best[0]):
            best = (gap, wall)
    return None if best is None else best[1]


def _resolve_openings(
    plan: BuildingPlan,
    walls: list[_Wall],
    merged_into: dict[str, str],
    cfg: dict[str, Any],
    report: SceneBuildReport,
) -> dict[str, tuple[_Wall, PlanOpening]]:
    """Assign every opening a host wall; bridge gaps / extend the wall so the opening fits."""
    ocfg = cfg["openings"]
    upm = plan.units_per_metre
    hosted: dict[str, tuple[_Wall, PlanOpening]] = {}
    by_id = {w.id: w for w in walls}
    removed: dict[str, str] = {}

    def live(wall_id: str | None) -> _Wall | None:
        while wall_id is not None and wall_id in removed:
            wall_id = removed[wall_id]
        if wall_id is None:
            return None
        wall_id = merged_into.get(wall_id, wall_id)
        while wall_id in removed:
            wall_id = removed[wall_id]
        return by_id.get(wall_id)

    def candidates_for(opening: PlanOpening, centre: np.ndarray, axis: np.ndarray, aligned: bool) -> list[tuple[float, _Wall]]:
        found: list[tuple[float, _Wall]] = []
        for wall in walls:
            angle = _angle_between(wall.direction, axis)
            if aligned and angle > ocfg["host_angle_deg"]:
                continue
            offset = abs(wall.offset(centre))
            if offset > wall.thickness / 2.0 + ocfg["host_offset_m"]:
                continue
            u = wall.along(centre)
            lo, hi = u - opening.width_m / 2.0, u + opening.width_m / 2.0
            # distance of the opening span to the wall interval (0 when overlapping)
            gap = max(lo - wall.length, 0.0 - hi, 0.0)
            if gap > ocfg["bridge_gap_m"] + 1e-9 and live(opening.host_wall_id) is not wall:
                continue
            overlap = max(0.0, min(hi, wall.length) - max(lo, 0.0))
            score = (live(opening.host_wall_id) is wall) * 0.5 + overlap / opening.width_m - offset - (0.0 if aligned else angle / 90.0)
            found.append((score, wall))
        found.sort(key=lambda item: -item[0])
        return found

    def try_host(opening: PlanOpening, centre: np.ndarray, candidates: list[tuple[float, _Wall]]) -> _Wall | None:
        host = candidates[0][1]
        u = host.along(centre)
        lo, hi = u - opening.width_m / 2.0, u + opening.width_m / 2.0
        if lo < -0.02 or hi > host.length + 0.02:
            # bridge collinear neighbours that touch the opening span, then extend if still short
            for _, other in candidates[1:]:
                if other is host or not _collinear(host, other, ocfg["host_angle_deg"], 1.0):
                    continue
                # Two pieces on either side of an opening are one wall only when their axes really coincide.  A thin partition
                # next to a thick exterior piece (offset 0.1-0.2 m) is a different wall: merging them skews the merged axis and
                # the solid then cuts into the neighbouring room.
                if abs(host.offset((other.p0 + other.p1) / 2.0)) > 0.5 * min(host.thickness, other.thickness):
                    continue
                d = host.direction
                a, b = sorted((float(np.dot(other.p0 - host.p0, d)), float(np.dot(other.p1 - host.p0, d))))
                if b < lo - ocfg["bridge_gap_m"] or a > hi + ocfg["bridge_gap_m"]:
                    continue
                # Two pieces with both faces measured and a clear step are a real reveal: the opening takes the host's faces.
                if host.evidence in _MEASURED_WALL_FACES and other.evidence in _MEASURED_WALL_FACES:
                    if _band_difference(host, other, (centre,)) >= ocfg["proven_step_m"]:
                        continue
                if _extension_blocked(host, min(0.0, a), max(host.length, b), walls, {id(other)}):
                    continue
                _merge_pair(host, other)
                removed[other.id] = host.id
                walls.remove(other)
                report.bridged += 1
            u = host.along(centre)
            lo, hi = u - opening.width_m / 2.0, u + opening.width_m / 2.0
            if lo < -0.02 or hi > host.length + 0.02:
                if _extension_blocked(host, min(0.0, lo), max(host.length, hi), walls, set()):
                    return None
                host.set_interval(min(0.0, lo), max(host.length, hi))
                report.openings_extended_wall += 1
        return host

    for opening in sorted(plan.openings, key=lambda o: o.id):
        report.openings_in += 1
        if opening.width_m < ocfg["min_width_m"]:
            report.openings_skipped += 1
            report.skipped_openings.append({"id": opening.id, "kind": opening.kind, "reason": f"width {opening.width_m:.2f} m below minimum {ocfg['min_width_m']} m"})
            report.warnings.append(f"opening {opening.id}: width {opening.width_m:.2f} m too small, skipped")
            continue
        centre = np.asarray(opening.center, dtype=float) / upm
        axis = np.array([math.cos(opening.axis_angle), math.sin(opening.axis_angle)])
        candidates = candidates_for(opening, centre, axis, True)
        reference: _Wall | None = None
        if not candidates:
            # the axis disagrees with the wall the opening sits on: trust the wall direction
            candidates = candidates_for(opening, centre, axis, False)
            if candidates:
                reference = candidates[0][1]
                report.openings_axis_overridden += 1
                report.warnings.append(
                    f"opening {opening.id}: axis disagrees with wall {reference.id} "
                    f"({_angle_between(reference.direction, axis):.0f} deg), wall direction used"
                )
        host = try_host(opening, centre, candidates) if candidates else None
        if host is None:
            reference = reference or (candidates[0][1] if candidates else _line_reference(opening, centre, axis, walls, cfg))
            made = _lintel_host(opening, centre, axis, reference, walls, plan, cfg, report)
            if made is None:
                report.openings_skipped += 1
                report.skipped_openings.append({"id": opening.id, "kind": opening.kind, "reason": "no host wall within reach"})
                report.warnings.append(f"opening {opening.id}: no host wall within reach, skipped")
                continue
            host, opening = made
            walls.append(host)
        hosted[opening.id] = (host, opening)
    # a later bridge may have merged the host of an earlier opening into another wall
    for key, (wall, opening) in list(hosted.items()):
        hosted[key] = (live(wall.id) or wall, opening)
    return hosted


def _clear_join_cut(
    opening: PlanOpening, position: float, width: float, geometry: Any, cfg: dict[str, Any], report: SceneBuildReport
) -> tuple[float, float]:
    """Move (<= ``max_join_shift_m``) or narrow an opening that reaches into the join-cut end of its host wall."""
    tolerance = 0.005
    lo, hi = position - width / 2.0, position + width / 2.0
    left, right = geometry.s_start - lo, hi - geometry.s_end
    if left <= tolerance and right <= tolerance:
        return position, width
    limit = float(cfg["openings"].get("max_join_shift_m", 0.10))
    available = float(geometry.s_end - geometry.s_start)
    if left > tolerance and right > tolerance:
        if available >= cfg["openings"]["min_width_m"]:
            report.openings_shifted += 1
            report.warnings.append(f"opening {opening.id}: narrowed {width:.2f} -> {available:.2f} m to clear the wall joins")
            return (geometry.s_start + geometry.s_end) / 2.0, available
    else:
        shift = left if left > tolerance else -right
        if abs(shift) <= limit:
            report.openings_shifted += 1
            report.warnings.append(f"opening {opening.id}: shifted {shift:+.3f} m along its wall to clear a join cut")
            return position + shift, width
    report.warnings.append(f"opening {opening.id}: lies in the join-cut region of its host wall (needs {max(left, right):.2f} m), kept in place")
    return position, width


def _inside_third_wall(point: np.ndarray, a: _Wall, b: _Wall, walls: list[_Wall]) -> bool:
    """True when the L-join corner ``point`` lies inside the body of a wall other than ``a`` / ``b``.

    Such a join would extend both ends through the third wall and stack their bodies on it (the end is
    better joined to that wall as a T instead).
    """
    for other in walls:
        if other is a or other is b:
            continue
        s = other.along(point)
        if 0.0 <= s <= other.length and abs(other.offset(point)) < other.thickness / 2.0 - 0.01:
            return True
    return False


def _build_joins(walls: list[_Wall], cfg: dict[str, Any]) -> list[tuple[str, str, str, str, str]]:
    """Return ``(type, wall_a, end_a, wall_b, end_b)`` tuples (T: b is the path wall, end_b ``path``)."""
    jcfg = cfg["walls"]["join"]
    reach = jcfg["reach_m"]
    candidates: list[tuple[int, float, str, _Wall, str, _Wall, str]] = []
    for ia, a in enumerate(walls):
        for b in walls[ia + 1:]:
            if _angle_between(a.direction, b.direction) < jcfg["min_angle_deg"]:
                continue
            da, db = a.direction, b.direction
            denominator = da[0] * db[1] - da[1] * db[0]
            if abs(denominator) < 1e-9:
                continue
            rel = b.p0 - a.p0
            t = (rel[0] * db[1] - rel[1] * db[0]) / denominator
            point = a.p0 + da * t
            sa, sb = a.along(point), b.along(point)
            info = []
            for wall, s, other in ((a, sa, b), (b, sb, a)):
                inside = 0.0 <= s <= wall.length
                start_dist, end_dist = abs(s), abs(s - wall.length)
                end = "start" if start_dist <= end_dist else "end"
                dist = min(start_dist, end_dist)
                end_zone = other.thickness / 2.0 + 0.03
                touching = (dist <= reach) if not inside else (dist <= end_zone)
                info.append((touching, end, dist, inside))
            (ta, ea, dista, _), (tb, eb, distb, _) = info
            corner_blocked = _inside_third_wall(point, a, b, walls)
            if ta and tb and not corner_blocked:
                candidates.append((0, dista + distb, "L", a, ea, b, eb))
            # A T-join whose end must be EXTENDED to a corner inside a third wall would run through that wall: the decision
            # follows the geometry, not the reach threshold (a lintel end touching the next wall piece was T-joined through it
            # to the wall behind at 0.398 m and left alone at 0.402 m reach).  An end already reaching into the path wall is kept.
            if ta and 0.0 <= sb <= b.length and not (corner_blocked and dista > b.thickness / 2.0):
                candidates.append((1, dista, "T", a, ea, b, "path"))
            if tb and 0.0 <= sa <= a.length and not (corner_blocked and distb > a.thickness / 2.0):
                candidates.append((1, distb, "T", b, eb, a, "path"))
    candidates.sort(key=lambda item: (item[0], item[1], item[3].id, item[5].id))
    used: set[tuple[str, str]] = set()
    joins: list[tuple[str, str, str, str, str]] = []
    for _, _, kind, a, end_a, b, end_b in candidates:
        keys = [(a.id, end_a)] + ([(b.id, end_b)] if kind == "L" else [])
        if any(key in used for key in keys):
            continue
        used.update(keys)
        joins.append((kind, a.id, end_a, b.id, end_b))
    return joins


def door_hand(hinge_side: str | None, swing_side: int | None) -> tuple[DoorOperation, Side]:
    """IFC4 single-swing operation and hinge side of a door from its measured hinge and swing.

    ``hinge_side`` is ``left`` when the hinge is at the ``-axis`` end of the opening and ``swing_side`` is +1 when the leaf
    swings to the ``+normal`` side (plan model).  IFC4 names the hand as seen from the side the door opens towards: that
    observer faces the wall, his left is the ``swing_side * axis`` direction, so the hinge (``left`` = the ``-axis`` end) is
    on his left exactly when ``hinge_side == "left"`` and ``swing_side == -1``, or ``hinge_side == "right"`` and
    ``swing_side == +1``.  Without a measured swing the hinge side is taken as given (its observer is unknown); without a
    hinge the default is a right-hinged door.
    """
    if hinge_side in ("left", "right") and swing_side in (1, -1):
        hinge_on_left = (hinge_side == "left") == (swing_side < 0)
    else:
        hinge_on_left = hinge_side == "left"
    return (DoorOperation.SINGLE_SWING_LEFT, Side.LEFT) if hinge_on_left else (DoorOperation.SINGLE_SWING_RIGHT, Side.RIGHT)


# --------------------------------------------------------------------------------------
# main entry
# --------------------------------------------------------------------------------------

def build_scene(
    plan: BuildingPlan,
    instances: dict[int, dict[str, Any]] | None = None,
    config: dict[str, Any] | None = None,
) -> Scene:
    """Build the IFC scene of one level.  The build report is in ``scene.derived["samcloud_build"]``.

    ``instances`` optionally maps an instance id to extra data (``confidence``, ``label``) that fills
    gaps of the plan's furniture items.  ``config`` overrides entries of :func:`load_scene_config`.
    """
    cfg = load_scene_config(config)
    mapping = load_furniture_mapping()
    report = SceneBuildReport()
    upm = plan.units_per_metre or 1.0

    # --- storey -------------------------------------------------------------------
    elevation = float(plan.floor_z) / upm
    if plan.ceiling_z is not None and plan.ceiling_z > plan.floor_z:
        height = (float(plan.ceiling_z) - float(plan.floor_z)) / upm
    else:
        height = float(cfg["storey"]["fallback_height_m"])
        report.warnings.append(f"storey height unknown, {height} m used")
    scene = Scene(
        settings=ProjectSettings(
            data_source=DataSource.PHOTOGRAMMETRY,
            indoor_detail=IndoorDetail.STRUCTURE_AND_FURNITURE,
            ceiling_ifc_class=CeilingIfcClass.COVERING,
        )
    )
    site = Site(id=_uid(plan, "site", "0"), name=plan.project_name)
    building = Building(id=_uid(plan, "building", "0"), name=plan.project_name, site_id=site.id)
    storey = Storey(
        id=_uid(plan, "storey", plan.level_id),
        name=plan.level_name or plan.level_id,
        elevation=elevation,
        height=height,
        building_id=building.id,
    )
    scene.sites[site.id] = site
    scene.buildings[building.id] = building
    scene.storeys[storey.id] = storey

    # --- walls --------------------------------------------------------------------
    walls = _plan_walls(plan, cfg, report)
    report.walls_in = len(walls)
    walls, merged_into = _dedup(walls, cfg, report, frozenset(o.host_wall_id for o in plan.openings if o.host_wall_id))
    walls = _trim_overlaps(walls, cfg, report)
    walls = _split_crossings(walls, cfg, report)
    merged_into = {k: v for k, v in merged_into.items()}
    hosted = _resolve_openings(plan, walls, merged_into, cfg, report)
    walls = _bridge_exterior_t_gaps(walls, plan.openings, upm, cfg, report, merged_into)
    live_walls = {wall.id: wall for wall in walls}
    for key, (wall, opening) in list(hosted.items()):
        target = wall.id
        while target in merged_into:
            target = merged_into[target]
        hosted[key] = (live_walls.get(target, wall), opening)
    walls = _trim_overlaps(walls, cfg, report, {id(w) for w, _ in hosted.values()})
    joins = _build_joins(walls, cfg)

    wall_ids: dict[str, str] = {}
    for wall in sorted(walls, key=lambda w: w.id):
        wall_id = _uid(plan, "wall", wall.id)
        wall_ids[wall.id] = wall_id
        function = WallFunction.EXTERIOR if wall.kind == "exterior" else WallFunction.INTERIOR
        notes = []
        if len(wall.members) > 1:
            notes.append("merged plan pieces: " + ", ".join(wall.members))
        if wall.single_line:
            notes.append("single line piece: default thickness")
        scene.add(
            WallElement(
                id=wall_id,
                name=wall.id,
                tag=wall.id,
                storey_id=storey.id,
                building_id=building.id,
                params=WallParams(
                    start=(float(wall.p0[0]), float(wall.p0[1])),
                    end=(float(wall.p1[0]), float(wall.p1[1])),
                    thickness=float(wall.thickness),
                    height=float(wall.height or height),
                    function=function,
                ),
                param_origin={
                    "start": ParamOrigin.FITTED,
                    "end": ParamOrigin.FITTED,
                    "thickness": ParamOrigin.ASSUMED if wall.single_line else ParamOrigin.MEASURED,
                    "height": ParamOrigin.MEASURED if wall.height and wall.height_source in _MEASURED_WALL_HEIGHTS else ParamOrigin.ASSUMED,
                },
                notes=notes,
            )
        )
    report.walls_out = len(walls)
    for kind, a_id, end_a, b_id, end_b in joins:
        scene.add(
            WallJoinElement(
                id=_uid(plan, "join", f"{a_id}/{end_a}/{b_id}/{end_b}"),
                name=f"{a_id}-{b_id}",
                storey_id=storey.id,
                params=WallJoinParams(
                    wall_a=wall_ids[a_id],
                    wall_b=wall_ids[b_id],
                    join_type=JoinType(kind),
                    end_a=JoinEnd(end_a),
                    end_b=JoinEnd(end_b),
                ),
            )
        )
        if kind == "L":
            report.joins_l += 1
        else:
            report.joins_t += 1

    for _attempt in range(3):
        wall_elements = [e for e in scene.elements.values() if isinstance(e, WallElement)]
        join_elements = [e for e in scene.elements.values() if isinstance(e, WallJoinElement)]
        geometries, _ = compute_wall_geometries(wall_elements, join_elements, {storey.id: elevation})
        # stubs that vanish in their join cuts cannot be exported: drop them (with their joins) and rebuild
        degenerate = {key for key, g in geometries.items() if any("no positive extent" in issue for issue in g.issues)}
        if not degenerate:
            break
        for key in sorted(degenerate):
            report.warnings.append(f"wall {scene.elements[key].name}: shorter than its join cuts, dropped")
            del scene.elements[key]
            report.walls_out -= 1
        for key in [k for k, e in scene.elements.items() if isinstance(e, WallJoinElement)
                    and (e.params.wall_a in degenerate or e.params.wall_b in degenerate)]:
            del scene.elements[key]
    wall_ids = {name: key for name, key in wall_ids.items() if key in scene.elements}
    geometries = {key: g for key, g in geometries.items() if key in scene.elements}
    footprints = {key: Polygon(g.footprint) for key, g in geometries.items()}
    for key, g in geometries.items():
        for issue in g.issues:
            report.warnings.append(f"wall {scene.elements[key].name}: {issue}")
    wall_union = unary_union(list(footprints.values())) if footprints else Polygon()

    # wall overlap QA (after joins)
    limit = cfg["qa"]["max_wall_overlap_share"]
    names = {key: scene.elements[key].name for key in footprints}
    keys = sorted(footprints)
    for i, ka in enumerate(keys):
        for kb in keys[i + 1:]:
            fa, fb = footprints[ka], footprints[kb]
            if not fa.intersects(fb):
                continue
            area = fa.intersection(fb).area
            share = area / max(min(fa.area, fb.area), 1e-12)
            if share > limit:
                report.wall_overlaps.append(
                    {"a": names[ka], "b": names[kb], "area_m2": round(area, 5), "share_of_smaller": round(share, 4)}
                )

    # --- openings -----------------------------------------------------------------
    reveals: dict[str, Polygon] = {}  # door / passage: the stretch of its host between both faces
    lintels: list[Polygon] = []  # ... those with wall above them up to the top
    for opening_id in sorted(hosted):
        wall, opening = hosted[opening_id]
        centre = np.asarray(opening.center, dtype=float) / upm
        wall_height = float(wall.height or height)
        position = wall.along(centre)
        width = float(opening.width_m)
        if wall.id not in wall_ids:
            report.openings_skipped += 1
            report.skipped_openings.append({"id": opening.id, "kind": opening.kind, "reason": f"host wall {wall.id} vanished in its join cuts"})
            report.warnings.append(f"opening {opening.id}: host wall {wall.id} vanished in its join cuts, skipped")
            continue
        trimmed = geometries.get(wall_ids[wall.id])
        if trimmed is not None:
            position, width = _clear_join_cut(opening, position, width, trimmed, cfg, report)
        if opening.kind == "window":
            sill = max(0.0, float(opening.sill_m))
        else:
            sill = 0.0
        # A passage is open up to the ceiling (measured: no point up to the ceiling): no lintel is built above it.  A door keeps its measured
        # (or default) head; the lintel above it exists where the points showed a head strip / wall above the opening.
        top = wall_height if opening.kind == "passage" else min(float(opening.head_m), wall_height)
        op_height = top - sill
        if op_height < 0.1:
            report.warnings.append(f"opening {opening.id}: head {opening.head_m} m below sill, skipped")
            report.openings_skipped += 1
            continue
        scene_open_id = _uid(plan, "opening", opening.id)
        host_id = wall_ids[wall.id]
        props: dict[str, PropertyValue] = {
            "PlanOpeningId": PropertyValue(value=opening.id, value_type="text"),
            "HeightSource": PropertyValue(value=str(opening.height_source), value_type="text"),
            "Evidence": PropertyValue(value=str(opening.evidence), value_type="text"),
        }
        if opening.kind == "window":
            props["MullionCount"] = PropertyValue(value=float(opening.mullion_count), value_type="number")
        if opening.confidence is not None:
            props["MeasuredConfidence"] = PropertyValue(value=float(opening.confidence), value_type="number")
        if opening.kind in ("door", "window"):
            props["Glazed"] = PropertyValue(value=bool(opening.glazed), value_type="boolean")
        label = {"door": "Door", "window": "Window"}.get(opening.kind, "Passage")
        scene.add(
            OpeningElement(
                id=scene_open_id,
                name=f"{label} opening {opening.id}",
                tag=opening.id,
                storey_id=storey.id,
                building_id=building.id,
                params=OpeningParams(
                    host_wall_id=host_id,
                    position_along_axis=max(position, width / 2.0),
                    width=width,
                    height=float(op_height),
                    sill_height=float(sill),
                ),
                param_origin=_opening_origins(opening),
            )
        )
        if opening.kind == "door":
            operation, side = door_hand(opening.hinge_side, opening.swing_side)
            door_props = dict(props)
            if opening.hinge_side:
                door_props["HingeSide"] = PropertyValue(value=str(opening.hinge_side), value_type="text")
            if opening.swing_side in (1, -1):
                door_props["SwingNormalSide"] = PropertyValue(value="+normal" if opening.swing_side > 0 else "-normal", value_type="text")
            if opening.leaf_angle_deg is not None:
                door_props["LeafOpeningAngleDeg"] = PropertyValue(value=float(opening.leaf_angle_deg), value_type="number")
            scene.add(
                DoorElement(
                    id=_uid(plan, "door", opening.id),
                    name=f"Door {opening.id}",
                    tag=opening.id,
                    storey_id=storey.id,
                    building_id=building.id,
                    params=DoorParams(opening_id=scene_open_id, operation=operation, swing_side=side, glazed=bool(opening.glazed)),
                    param_origin={
                        "operation": ParamOrigin.MEASURED if opening.hinge_side else ParamOrigin.ASSUMED,
                        "swing_side": ParamOrigin.MEASURED if opening.swing_side else ParamOrigin.ASSUMED,
                    },
                    properties=door_props,
                )
            )
        elif opening.kind == "window":
            scene.add(
                WindowElement(
                    id=_uid(plan, "window", opening.id),
                    name=f"Window {opening.id}",
                    tag=opening.id,
                    storey_id=storey.id,
                    building_id=building.id,
                    params=WindowParams(opening_id=scene_open_id, mullion_positions=_mullion_positions(opening, wall, width)),
                    param_origin={"operation": ParamOrigin.ASSUMED},
                    properties=props,
                )
            )
        report.openings_hosted += 1
        if opening.kind in ("door", "passage") and trimmed is not None:
            placed = max(position, width / 2.0)
            first, last = placed - width / 2.0, placed + width / 2.0
            right, left = trimmed.offset_right - 1e-6, trimmed.offset_left + 1e-6  # (1 micrometre: no hairlines along the faces)
            reveals[opening.id] = Polygon([trimmed.point(first, right), trimmed.point(last, right), trimmed.point(last, left), trimmed.point(first, left)])
            if top < wall_height - 1e-3:
                lintels.append(reveals[opening.id])

    # --- spaces + ceilings --------------------------------------------------------
    room_polygons: list[tuple[str, Polygon, Polygon]] = []  # (space id, clipped, original)
    # A doorway is no wall at floor level: the plan rooms include their half of each door reveal, so the spaces are clipped by
    # the walls with the door / passage voids taken out (each room ends at the door line; a room reached across a doorway stays one polygon).
    doorways = []
    for opening in plan.openings:
        if opening.kind not in ("door", "passage"):
            continue
        if opening.id in reveals:
            doorways.append(reveals[opening.id])
            continue
        axis = np.array([math.cos(opening.axis_angle), math.sin(opening.axis_angle)])
        normal = np.array([-axis[1], axis[0]])
        centre = np.asarray(opening.center, dtype=float) / upm
        half_w, half_t = float(opening.width_m) / 2.0, max(float(opening.wall_thickness_m), 0.05) / 2.0
        doorways.append(Polygon([tuple(centre + axis * a + normal * b) for a, b in ((-half_w, -half_t), (half_w, -half_t), (half_w, half_t), (-half_w, half_t))]))
    room_wall_union = wall_union.difference(unary_union(doorways)) if doorways and not wall_union.is_empty else wall_union
    lintel_union = unary_union(lintels) if lintels else Polygon()
    # walls that reach the ceiling (a half-height wall or a pillar hole keeps the ceiling above it)
    reach_top = height - float(cfg["ceiling"]["thickness_m"])
    tall_walls = unary_union([footprints[key] for key in footprints if geometries[key].height >= reach_top])
    for room in sorted(plan.rooms, key=lambda r: r.id):
        exterior = np.asarray(room.polygon["exterior"], dtype=float) / upm
        holes = [np.asarray(h, dtype=float) / upm for h in room.polygon.get("holes", [])]
        original = _clean_polygon(Polygon(exterior, holes))
        if original is None:
            report.warnings.append(f"room {room.id} {room.name!r}: empty polygon, skipped")
            report.room_areas.append({"id": room.id, "name": room.name, "plan_m2": float(room.area_m2), "ifc_m2": 0.0})
            continue
        clipped_all = _polygons(original.difference(room_wall_union)) if not room_wall_union.is_empty else [original]
        clipped_all = sorted(
            [cleaned for piece in clipped_all if (cleaned := _clean_polygon(piece)) is not None and cleaned.area >= 0.05],
            key=lambda p: -p.area,
        )
        if not clipped_all:
            report.warnings.append(f"room {room.id} {room.name!r}: nothing left after wall clipping, skipped")
            report.room_areas.append({"id": room.id, "name": room.name, "plan_m2": float(room.area_m2), "ifc_m2": 0.0})
            continue
        # The wall footprints can cut a plan room into several pieces: every piece of at least 0.3 m2 becomes a space of its own
        # (same name, tag "R003.1"), so no measured floor is silently dropped; slivers below that are lost (and reported).
        kept_pieces = [clipped_all[0]] + [piece for piece in clipped_all[1:] if piece.area >= 0.3]
        lost = original.area - sum(p.area for p in kept_pieces)
        if len(clipped_all) > 1:
            report.warnings.append(
                f"room {room.id} {room.name!r}: wall clipping split it into {len(clipped_all)} pieces; {len(kept_pieces)} kept as spaces"
            )
        report.room_areas.append({"id": room.id, "name": room.name, "plan_m2": float(room.area_m2), "ifc_m2": 0.0})
        for piece_index, piece in enumerate(kept_pieces):
            clipped = _clean_polygon(piece.simplify(1e-4, preserve_topology=True)) or piece
            piece_key = room.id if piece_index == 0 else f"{room.id}.{piece_index}"
            space_id = _uid(plan, "space", piece_key)
            space = SpaceElement(
                id=space_id,
                name=room.name,
                tag=piece_key,
                storey_id=storey.id,
                building_id=building.id,
                params=SpaceParams(
                    name=room.name,
                    outline=_ring(clipped),
                    holes=_interior_rings(clipped, min_area_m2=_MIN_INTERIOR_AREA_M2),
                ),
                param_origin={"name": ParamOrigin.MEASURED, "outline": ParamOrigin.FITTED},
                properties={
                    "PlanArea": PropertyValue(value=float(room.area_m2), value_type="number"),
                    "PlanRoomId": PropertyValue(value=room.id, value_type="text"),
                    **space_height_properties(room, upm, elevation),
                },
            )
            scene.add(space)
            room_polygons.append((space_id, clipped, original))
            report.spaces += 1
            report.room_area_plan_m2 += float(room.area_m2) if piece_index == 0 else 0.0
            report.room_area_scene_m2 += float(clipped.area)
            report.room_areas[-1]["ifc_m2"] += float(clipped.area)
            if lost > 0.02 * max(original.area, 1e-9):
                report.warnings.append(f"room {room.id} {room.name!r}: {lost:.2f} m2 lost to wall clipping")

            # A hole with a full-height wall in it is an island of walls (or a room inside this one): no ceiling of this room there.
            ceiling_shell = Polygon(clipped.exterior, [hole for hole in clipped.interiors if Polygon(hole).intersection(tall_walls).area > 1e-4])
            # The floor runs through a door reveal, the ceiling ends at the wall above the door.
            shells = [ceiling_shell]
            if ceiling_shell.interiors:
                shells = _without_holes(ceiling_shell)
            if not lintel_union.is_empty and ceiling_shell.intersects(lintel_union):
                # Room outline and lintel share their edges: opened by 0.1 mm, so no hairline spike of either is left in the outline.
                trimmed_shell = ceiling_shell.difference(lintel_union).buffer(-1e-4, join_style=2).buffer(1e-4, join_style=2).simplify(1e-5, preserve_topology=True)
                shells = [piece for part in _polygons(make_valid(trimmed_shell)) for piece in _without_holes(part) if piece.area >= 0.05]
            # Measured per-room ceiling (flat, sloped plane, or two planes split at the ridge / valley), else the storey ceiling.
            for ceiling_index, (ceiling_piece, ceiling_elevation, plane_normal, plane_offset, ceiling_measured) in enumerate(
                piece_of for shell in shells for piece_of in room_ceiling_pieces(room, upm, elevation + height, shell)
            ):
                ceiling_key = piece_key if ceiling_index == 0 else f"{piece_key}/c{ceiling_index + 1}"
                scene.add(
                    CeilingElement(
                        id=_uid(plan, "ceiling", ceiling_key),
                        name=(f"Ceiling {room.name}" if room.name else f"Ceiling {room.id}") + ("" if ceiling_index == 0 else f" {ceiling_index + 1}"),
                        tag=ceiling_key,
                        storey_id=storey.id,
                        building_id=building.id,
                        space_ids=[space_id],
                        params=CeilingParams(
                            outline=_ring(ceiling_piece),
                            elevation=ceiling_elevation,
                            plane_normal=plane_normal,
                            plane_offset=plane_offset,
                            thickness=float(cfg["ceiling"]["thickness_m"]),
                            trim_to_walls=False,
                        ),
                        param_origin={"outline": ParamOrigin.FITTED, "elevation": ParamOrigin.MEASURED if ceiling_measured else ParamOrigin.ASSUMED, "thickness": ParamOrigin.ASSUMED},
                    )
                )
            report.ceilings += 1
            overlap = unary_union(shells).intersection(room_wall_union).area if shells and not room_wall_union.is_empty else 0.0
            report.ceiling_wall_overlap_max_m2 = max(report.ceiling_wall_overlap_max_m2, float(overlap))

    # --- floor slabs --------------------------------------------------------------
    slab_cfg = cfg["slab"]
    if room_polygons:
        gap = float(slab_cfg["close_gap_m"])
        union = unary_union([original for _, _, original in room_polygons] + list(footprints.values()))
        union = union.buffer(gap, join_style=2, mitre_limit=5.0).buffer(-gap, join_style=2, mitre_limit=5.0)
        room_union = unary_union([clipped for _, clipped, _ in room_polygons])
        index = 0
        for component in sorted(_polygons(union), key=lambda p: (p.bounds[0], p.bounds[1])):
            if component.area < slab_cfg["min_area_m2"] or not component.intersects(room_union):
                continue
            simple = _clean_polygon(component.simplify(float(slab_cfg["simplify_m"]), preserve_topology=True)) or component
            holes = _interior_rings(simple, min_area_m2=float(slab_cfg["min_hole_area_m2"]))
            index += 1
            scene.add(
                SlabElement(
                    id=_uid(plan, "slab", str(index)),
                    name=f"Floor slab {index}",
                    tag=f"S{index:02d}",
                    storey_id=storey.id,
                    building_id=building.id,
                    params=SlabParams(
                        outline=_ring(simple),
                        thickness=float(slab_cfg["thickness_m"]),
                        top_elevation=elevation,
                        openings=holes,
                    ),
                    param_origin={"outline": ParamOrigin.FITTED, "thickness": ParamOrigin.ASSUMED, "top_elevation": ParamOrigin.MEASURED},
                )
            )
            report.slabs += 1
    else:
        report.warnings.append("no rooms: no floor slab built")

    # --- furniture ----------------------------------------------------------------
    _add_furniture(plan, scene, storey, building.id, room_polygons, mapping, cfg, report, instances or {}, elevation, height)
    # --- objects on the ceiling -----------------------------------------------------
    scene.derived["samcloud_ceiling_objects"] = add_ceiling_objects(
        plan, scene, storey, building.id, lambda key: _uid(plan, "ceiling-object", key), room_polygons, mapping, report.warnings,
    )
    # --- measured structure: columns, beams, stairs ---------------------------------
    structure_counts = add_structure_elements(plan, scene, storey, building.id, lambda kind, key: _uid(plan, kind, key), elevation, height, report.warnings)
    scene.derived["samcloud_structure"] = structure_counts

    scene.derived["samcloud_frame"] = {
        "UnitsPerMetre": float(plan.units_per_metre),
        "ScaleFactor": float(plan.frame.get("scale_factor", 1.0)),
        "Origin": plan.frame.get("origin"),
        "Rotation": plan.frame.get("rotation"),
        "LevelId": plan.level_id,
        "LevelName": plan.level_name,
        "ProjectName": plan.project_name,
        "Georeferenced": "no",
    }
    scene.derived["samcloud_build"] = asdict(report)
    return scene


def _add_furniture(
    plan: BuildingPlan,
    scene: Scene,
    storey: Storey,
    building_id: str,
    room_polygons: list[tuple[str, Polygon, Polygon]],
    mapping: FurnitureMapping,
    cfg: dict[str, Any],
    report: SceneBuildReport,
    instances: dict[int, dict[str, Any]],
    elevation: float,
    height: float,
) -> None:
    upm = plan.units_per_metre
    fcfg = cfg["furniture"]
    counters: dict[str, int] = {}
    for item in sorted(plan.furniture, key=lambda f: f.id):
        report.furniture_in += 1
        polygon = _furniture_polygon(item.footprint, upm, float(fcfg["simplify_m"]))
        if polygon is None or polygon.area < fcfg["min_area_m2"]:
            report.warnings.append(f"furniture {item.id} {item.label!r}: unusable footprint, skipped")
            continue
        extra_info = instances.get(item.instance_id, {}) if item.instance_id is not None else {}
        confidence = item.confidence if item.confidence is not None else extra_info.get("confidence")
        klass, object_type = mapping.resolve(
            label=item.label,
            semantic_label=item.semantic_label,
            catalogue_kind=item.catalogue_kind,
            layer=item.layer,
        )
        # heights relative to the storey (m)
        if item.z_source in ("instance_bbox", "points") and item.z_min is not None and item.z_max is not None:
            z0 = float(item.z_min) / upm - elevation
            z1 = float(item.z_max) / upm - elevation
        elif item.default_z_m:
            z0, z1 = float(item.default_z_m[0]), float(item.default_z_m[1])
        else:
            z0 = 0.0
            z1 = float(item.default_height_m or mapping.fallback_height_m)
        # Depth evidence of furniture voxels at floor level inside the footprint says the object stands on the floor (a shelf unit beside a wall).
        if item.mounting:  # decided by the evidence of the scan, not by the name
            hanging = item.mounting != "standing"
        else:
            hanging = _is_wall_mounted_furniture(item) and _furniture_near_wall(polygon, plan, upm) and item.floor_contact is not True
        elevated_component = (item.catalogue_kind or "").lower() in {"worktop", "kitchen_sink", "cooktop"}
        # Scan bboxes often start at the visible tabletop, mattress or fridge
        # front.  Everything that can stand therefore starts on the floor; only
        # verified wall-mounted objects retain a measured suspended height.
        if not hanging and not elevated_component:
            z0 = 0.0
        z1 = min(z1, height)
        if z1 - z0 < 0.02:
            report.warnings.append(f"furniture {item.id} {item.label!r}: height {z1 - z0:.3f} m, skipped")
            continue
        centroid = polygon.centroid
        space_id = None
        for sid, clipped, original in room_polygons:
            if original.contains(centroid) or clipped.contains(centroid):
                space_id = sid
                break
        if space_id is None and room_polygons:
            nearest = min(room_polygons, key=lambda r: r[2].distance(centroid))
            if nearest[2].distance(centroid) <= 0.3:
                space_id = nearest[0]
        label = (item.label or klass.key).strip()
        counters[label] = counters.get(label, 0) + 1
        solid_parts = _furniture_parts(item, upm, elevation, polygon, z0, z1, hanging or elevated_component, float(fcfg["min_area_m2"]), float(fcfg["simplify_m"]))
        scene.add(
            FurnitureElement(
                id=_uid(plan, "furniture", item.id),
                name=f"{label} {counters[label]:02d}",
                tag=item.id,
                ifc_class=klass.ifc_class,
                predefined_type=klass.predefined_type,
                storey_id=storey.id,
                building_id=building_id,
                space_ids=[space_id] if space_id else [],
                params=OpaqueParams(
                    footprint=[[float(x), float(y)] for x, y in _ring(polygon)],
                    z_bottom=elevation + z0,
                    z_top=elevation + z1,
                    parts=solid_parts,
                    mounting="resting" if item.mounting == "resting" else "hanging" if hanging else "grounded",
                    object_type=object_type,
                    colour=list(klass.colour),
                    label=item.label,
                    semantic_label=item.semantic_label,
                    instance_id=item.instance_id,
                    confidence=confidence,
                    z_source=item.z_source,
                    layer=item.layer,
                    catalogue_kind=item.catalogue_kind,
                    class_key=klass.key,
                    facing_deg=item.facing_deg,
                    back_wall_id=item.back_wall_id,
                    shape_source=item.shape_source,
                    **_furniture_shape(item, upm, elevation + height),
                    # AI-assistant items without point-cloud evidence (samcloud.assistant): only then the key exists at all
                    **({"evidence": "manual"} if (item.fit or {}).get("evidence") == "manual" else {}),
                ),
            )
        )
        report.furniture_out += 1


_WALL_MOUNTED_FURNITURE = frozenset({
    "upper_cabinet", "wall cabinet", "washbasin", "lavabo", "wall shelf", "shelf", "hood", "wall tv", "tv", "television", "mirror cabinet",
})


def _is_wall_mounted_furniture(item: PlanFurniture) -> bool:
    """Whether the semantic kind is allowed to retain a suspended z origin."""
    names = (item.catalogue_kind or "", item.semantic_label or "", item.label or "")
    return item.layer == "A-OBEN" or any(str(name).strip().lower() in _WALL_MOUNTED_FURNITURE for name in names)


def _furniture_near_wall(polygon: Polygon, plan: BuildingPlan, upm: float, maximum_m: float = 0.30) -> bool:
    """Require a wall face within a small metric distance for hanging furniture."""
    if not plan.walls:
        return False
    limit = maximum_m
    for wall in plan.walls:
        for points in (wall.face_a, wall.face_b, wall.centreline):
            if not points or len(points) < 2:
                continue
            line = LineString(np.asarray(points, dtype=float) / upm)
            if line.length and polygon.distance(line) <= limit:
                return True
    return False


def _furniture_shape(item: PlanFurniture, upm: float, z_limit: float) -> dict[str, Any]:
    """Reconstructed 3D shape (samcloud.bim.furniture_recon) in scene metres: {"solids": [...]} or {"mesh": {...}}; {} for footprint bodies.

    Solids: {"type", "role", "part_source", "outline" (ring), "z_bottom", "z_top", "center"/"radius" for cylinders}; z absolute, capped at
    the storey top ``z_limit``.  Mesh: {"vertices", "faces"} (absolute metres)."""
    if item.shape_source == "mesh" and item.mesh and item.mesh.get("vertices") and item.mesh.get("faces"):
        vertices = np.asarray(item.mesh["vertices"], dtype=float) / upm
        vertices[:, 2] = np.minimum(vertices[:, 2], z_limit)
        return {"mesh": {"vertices": vertices.tolist(), "faces": [[int(v) for v in face] for face in item.mesh["faces"]]}}
    if item.shape_source != "parametric" or not item.solids:
        return {}
    solids: list[dict[str, Any]] = []
    for raw in item.solids:
        ring = raw.get("outline") or []
        if len(ring) < 3:
            continue
        bottom, top = float(raw["z_min"]) / upm, min(float(raw["z_max"]) / upm, z_limit)
        if top - bottom < 0.005:
            continue
        solid = {
            "type": str(raw.get("type") or "extrusion"), "role": str(raw.get("role") or ""), "part_source": str(raw.get("part_source") or "measured"),
            "outline": [[float(x) / upm, float(y) / upm] for x, y in ring], "z_bottom": bottom, "z_top": top,
        }
        if solid["type"] == "cylinder" and raw.get("center") is not None and raw.get("radius"):
            solid["center"] = [float(raw["center"][0]) / upm, float(raw["center"][1]) / upm]
            solid["radius"] = float(raw["radius"]) / upm
        solids.append(solid)
    return {"solids": solids} if solids else {}


def _furniture_parts(
    item: PlanFurniture, upm: float, elevation: float, fallback: Polygon, z0: float, z1: float, hanging: bool,
    min_area: float, simplify: float,
) -> list[dict[str, Any]]:
    """Convert optional local-plan stepped parts to scene metres, with safe fallback."""
    output: list[dict[str, Any]] = []
    for raw in item.parts:
        if not isinstance(raw, dict):
            continue
        polygon = _furniture_polygon(raw.get("footprint") or [], upm, simplify)
        if polygon is None or polygon.area < min_area:
            continue
        low = float(raw.get("z_min", item.z_min if item.z_min is not None else elevation * upm)) / upm - elevation
        high = float(raw.get("z_max", item.z_max if item.z_max is not None else z1 * upm)) / upm - elevation
        if not hanging:
            low = 0.0
        high = min(high, z1)
        if high - low < 0.02:
            continue
        output.append({"footprint": [[float(x), float(y)] for x, y in _ring(polygon)], "z_bottom": elevation + low, "z_top": elevation + high})
    if output:
        return output
    return [{"footprint": [[float(x), float(y)] for x, y in _ring(fallback)], "z_bottom": elevation + z0, "z_top": elevation + z1}]


def _mullion_positions(opening: PlanOpening, wall: "_Wall", width: float) -> list[float]:
    """Mullion centres in metres from the left edge along the host wall direction (the opening frame's x axis).

    ``PlanOpening.mullion_u`` is measured along the opening axis (``axis_angle``); when the wall runs the other
    way the positions are mirrored.
    """
    positions = [float(value) for value in opening.mullion_u if 0.0 < float(value) < width]
    if not positions:
        # Older plan models only know the count: spread the bars evenly.
        return [width * (index + 1) / (opening.mullion_count + 1) for index in range(max(int(opening.mullion_count), 0))]
    axis = np.array([math.cos(opening.axis_angle), math.sin(opening.axis_angle)])
    if float(np.dot(wall.direction, axis)) < 0.0:
        positions = [width - value for value in positions]
    return sorted(positions)


def _opening_origins(opening: PlanOpening) -> dict[str, ParamOrigin]:
    measured_height = opening.height_source != "default"
    height_origin = ParamOrigin.MEASURED if measured_height else ParamOrigin.ASSUMED
    return {
        "host_wall_id": ParamOrigin.FITTED,
        "position_along_axis": ParamOrigin.MEASURED,
        "width": ParamOrigin.MEASURED,
        "height": height_origin,
        "sill_height": height_origin,
    }


def _furniture_polygon(points: list[list[float]], upm: float, simplify: float) -> Polygon | None:
    if not points or len(points) < 3:
        return None
    polygon = Polygon(np.asarray(points, dtype=float) / upm)
    if not polygon.is_valid:
        pieces = sorted(_polygons(make_valid(polygon)), key=lambda p: -p.area)
        polygon = pieces[0] if pieces else polygon
    if polygon.is_empty:
        return None
    if simplify > 0:
        simple = polygon.simplify(simplify, preserve_topology=True)
        if isinstance(simple, Polygon) and not simple.is_empty and simple.is_valid and len(simple.exterior.coords) > 3:
            polygon = simple
    if not polygon.is_valid or polygon.is_empty or len(polygon.exterior.coords) < 4:
        rect = polygon.minimum_rotated_rectangle
        if not isinstance(rect, Polygon) or rect.is_empty:
            return None
        polygon = rect
    return Polygon(polygon.exterior)
