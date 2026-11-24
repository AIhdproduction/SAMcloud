"""Recessed built-in niches: an open wardrobe / coat niche set into a wall.

A niche is a recess behind a wall line: the room outline (or a wall piece)
stops at the *opening*, behind it stand tall contents (shelves, hanging coats,
boxes), and at least one side panel / wall stub runs from the opening back into
the recess.  The wall must follow the niche (opening edge -> side panel -> back
wall -> side panel) instead of running straight along the opening, and nothing
that lies behind the back wall belongs in the plan.

The back wall is searched where it is actually visible, strongest evidence first:

1. ``wall_line``    a measured A-WALL line parallel to the opening, adjacent to it;
2. ``wall_points``  a dense line of wall-class points across the niche (seen above
   the top shelf / between the contents / at the corners);
3. ``room_edge``    a parallel ceiling-derived room-outline edge beside the niche;
4. ``side_panel``   the far end of the side panels where they stop inside the recess;
5. ``contents``     the far boundary of the contents (robust percentile of their
   depth, not the maximum) - the niche is closed there and flagged as
   "no back wall seen".

An opening with free floor behind it is a doorway, and a wardrobe standing in
front of a flat wall has no recess behind the wall line; neither is a niche.
Every metric value is multiplied by ``units_per_metre``; no project constants.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Any, Iterable

import cv2
import numpy as np
from shapely.geometry import LineString, Point, Polygon

from .builtin_fronts import (
    EINB_ID_BASE,
    BuiltinFrontSettings,
    _angle_gap,
    _angle_of,
    _apply_strip,
    _build_chains,
    _Chain,
    _collect_segments,
    _extent_gap,
    _fill_fractions,
    _frame_point,
    _polygon_mask,
    _Segment,
    _strip_label,
    _strip_polygon,
    _tall_runs,
)
from .raster_grid import RasterGrid

NICHE_ID_BASE = EINB_ID_BASE + 500_000
NICHE_LABELS: dict[str, str] = {"Schrank": "Garderobe", "Einbau": "Einbauschrank"}


@dataclass(frozen=True, slots=True)
class NicheSettings:
    min_width_m: float = 0.6
    max_width_m: float = 3.0
    min_depth_m: float = 0.20
    max_depth_m: float = 1.20
    content_height_m: float = 1.0
    front_margin_m: float = 0.25
    row_coverage: float = 0.30
    row_gap_m: float = 0.10
    back_slack_m: float = 0.20
    min_run_coverage: float = 0.60
    evidence_tolerance_m: float = 0.10
    panel_reach_m: float = 0.30
    panel_forward_m: float = 0.20
    panel_min_length_m: float = 0.15
    wall_point_bin_m: float = 0.03
    wall_point_min_count: int = 15
    wall_point_min_extent_m: float = 0.35
    wall_point_end_margin_m: float = 0.08
    chain_adjacency_m: float = 0.30
    chain_min_support_m: float = 0.35
    beyond_reach_m: float = 0.60
    fill_fraction: float = 0.5
    room_free_fraction: float = 0.5
    room_reach_m: float = 0.5
    require_panel: bool = True
    max_front_wall_fraction: float = 0.30
    interior_wall_inset_m: float = 0.04
    interior_wall_min_m: float = 0.10
    fixture_reach_m: float = 0.30
    fixture_fraction: float = 0.03
    fixture_dilate_m: float = 0.15
    beyond_inset_m: float = 0.04


def content_mask(tall: np.ndarray, height_map: np.ndarray, units_per_metre: float, settings: NicheSettings = NicheSettings()) -> np.ndarray:
    """Pixels that hold niche contents: tall furniture or anything at least ``content_height_m`` high."""
    return np.asarray(tall, dtype=bool) | (np.asarray(height_map) >= settings.content_height_m * units_per_metre)


def resolve_recessed_niches(
    layers: dict[str, list[dict[str, Any]]],
    room_polygons: Iterable[Polygon],
    *,
    content: np.ndarray,
    free_floor: np.ndarray,
    fine_classes: np.ndarray,
    fine_names: list[str],
    grid: RasterGrid,
    units_per_metre: float,
    wall_points: np.ndarray | None = None,
    exclude: np.ndarray | None = None,
    settings: NicheSettings = NicheSettings(),
) -> dict[str, Any]:
    """Detect recessed niches, rewrite walls / A-EINB around them and clear what lies behind them.

    ``layers`` is modified in place.  The result lists every niche (``niches``) and
    the strips taken (``strips``, to be excluded from ``resolve_builtin_fronts``).
    ``exclude`` is a raster of sanitary fixture / kitchen pixels: they never count as contents and
    a recess beside them is not a niche.
    """
    upm = max(float(units_per_metre), 1e-9)
    stats: dict[str, Any] = {"niches": [], "strips": [], "niche_beyond_removed": 0}
    inner: dict[str, Any] = {"builtin_fronts": 0, "builtin_walls_added": 0, "builtin_walls_removed": 0, "builtin_furniture_absorbed": 0, "builtin_einb_absorbed": 0, "builtin_strips": []}
    walls = layers.setdefault("A-WALL", [])
    bf = BuiltinFrontSettings(
        min_length_m=settings.min_width_m, front_margin_m=settings.front_margin_m, front_reach_m=settings.max_depth_m,
        side_wall_reach_m=settings.panel_reach_m, run_end_fraction=0.0,  # niche contents (hanging coats) are sparse: no end trimming
    )
    segments = _collect_segments(walls, list(room_polygons), upm, bf)
    chains = _build_chains(segments, upm, bf)
    content_mask_ = np.asarray(content, dtype=bool)
    fixtures = np.asarray(exclude, dtype=bool) if exclude is not None else np.zeros_like(content_mask_)
    if fixtures.any():
        # Sanitary fixtures and kitchen units are tall too, but they are not niche contents.
        radius = max(1, int(round(settings.fixture_dilate_m * upm / grid.pixel_size)))
        content_mask_ = content_mask_ & ~(cv2.dilate(fixtures.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1,) * 2)) > 0)
    free = np.asarray(free_floor, dtype=bool)
    expand = cv2.dilate(content_mask_.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    points = None if wall_points is None or not len(wall_points) else np.asarray(wall_points, dtype=float)
    accepted: list[Polygon] = []
    for front in sorted(chains, key=lambda chain: -chain.length):
        if front.length < settings.min_width_m * upm or not any(member.source in {"wall", "room"} for member in front.members):
            continue
        for side in (1.0, -1.0):
            for run in _tall_runs(front, side, expand, grid, upm, bf):
                first = _contents_extent(front, side, run, content_mask_, grid, upm, settings)
                if first is None:
                    continue
                # Contents that stand in front of the measured opening line move the opening forward to them.
                shift = -first[0] if -first[0] >= 0.05 * upm else 0.0
                line = replace(front, offset=front.offset - side * shift) if shift > 0.0 else front
                niche = _plan_niche(line, side, run, chains, segments, content_mask_, free, points, grid, upm, settings)
                if niche is None:
                    continue
                span, depth, evidence, info = niche
                # A measured wall along the whole front is a wall with a unit behind it, not an opening.
                if _wall_cover(line, span) > settings.max_front_wall_fraction:
                    continue
                strip = _strip_polygon(line, side, span, depth)
                if any(strip.intersection(other).area > 0.30 * strip.area for other in accepted):
                    continue
                # A measured wall that crosses the recess (wall stub, door frame) means the recess is a passage or jog.
                inner_strip = strip.buffer(-settings.interior_wall_inset_m * upm)
                if not inner_strip.is_empty and any(
                    entity.get("type") == "polyline" and len(entity.get("points", [])) == 2 and not entity.get("closed")
                    and LineString(entity["points"]).intersection(inner_strip).length >= settings.interior_wall_min_m * upm
                    for entity in layers["A-WALL"]
                ):
                    continue
                # A recess next to a toilet / sink / kitchen unit is a service shaft or kitchen run, not a niche.
                near = _polygon_mask(strip.buffer(settings.fixture_reach_m * upm), grid)
                if near.any() and float(np.mean(fixtures[near])) >= settings.fixture_fraction:
                    continue
                fill = _fill_fractions(strip, content_mask_, free, grid)
                if fill["fill"] < settings.fill_fraction or fill["free"] > 1.0 - settings.fill_fraction:
                    continue
                room = _fill_fractions(_strip_polygon(line, -side, span, settings.room_reach_m * upm), content_mask_, free, grid)
                if room["free"] < settings.room_free_fraction or room["free"] < fill["free"] + 0.30:
                    continue
                label = NICHE_LABELS.get(_strip_label(strip, fine_classes, fine_names, grid), _strip_label(strip, fine_classes, fine_names, grid))
                identifier = NICHE_ID_BASE + len(stats["niches"])
                _apply_strip(layers, inner, line, None, side, span, depth, strip, fine_classes, fine_names, grid, upm, bf, kind="niche", label=label, identifier=identifier)
                inner["builtin_fronts"] += 1
                _close_side_panels(layers, line, side, span, depth, upm)
                removed = _clear_beyond(layers, line, side, span, depth, upm, settings)
                stats["niche_beyond_removed"] += removed
                accepted.append(strip)
                stats["strips"].append(strip)
                stats["niches"].append({
                    "opening_m": (span[1] - span[0]) / upm, "depth_m": depth / upm, "evidence": evidence, "no_back_wall_seen": evidence == "contents",
                    "label": label, "removed_beyond": removed, "bounds": [float(value) for value in strip.bounds], **info,
                })
    stats["niche_count"] = len(stats["niches"])
    stats["niche_walls_added"] = inner["builtin_walls_added"]
    stats["niche_furniture_absorbed"] = inner["builtin_furniture_absorbed"]
    return stats


def _wall_cover(front: _Chain, span: tuple[float, float]) -> float:
    """Fraction of ``span`` that measured wall pieces (not room-outline edges) cover along the front."""
    intervals = []
    for member in front.members:
        if member.source != "wall":
            continue
        projections = sorted(float(front.unit @ np.asarray(point)) for point in (member.line.coords[0], member.line.coords[-1]))
        low, high = max(projections[0], span[0]), min(projections[1], span[1])
        if high > low:
            intervals.append((low, high))
    covered, cursor = 0.0, span[0]
    for low, high in sorted(intervals):
        low = max(low, cursor)
        if high > low:
            covered += high - low
            cursor = high
    return covered / max(span[1] - span[0], 1e-9)


# -- planning ---------------------------------------------------------------

def _hit_grid(front: _Chain, side: float, span: tuple[float, float], mask: np.ndarray, grid: RasterGrid, upm: float, reach: float, margin: float) -> tuple[np.ndarray, np.ndarray]:
    """Boolean (along samples x depths) array of mask pixels in the region behind the front, and the depths."""
    step = max(grid.pixel_size, 0.02 * upm)
    samples = np.arange(span[0], span[1] + 1e-9, step)
    depths = np.arange(-margin, reach + 1e-9, grid.pixel_size)
    if len(samples) == 0:
        return np.zeros((0, len(depths)), dtype=bool), depths
    along = np.repeat(samples, len(depths))
    across = np.tile(depths, len(samples))
    points = _frame_point(front, side, along, across)
    rows = np.rint((grid.max_y - points[:, 1]) / grid.pixel_size - 0.5).astype(int)
    cols = np.rint((points[:, 0] - grid.min_x) / grid.pixel_size - 0.5).astype(int)
    valid = (rows >= 0) & (rows < grid.height) & (cols >= 0) & (cols < grid.width)
    hit = np.zeros(len(points), dtype=bool)
    hit[valid] = mask[rows[valid], cols[valid]]
    return hit.reshape(len(samples), len(depths)), depths


def _contents_extent(front: _Chain, side: float, run: tuple[float, float], content: np.ndarray, grid: RasterGrid, upm: float, settings: NicheSettings) -> tuple[float, float, float] | None:
    """(near, far, coverage) of the contents across the run, in front coordinates.

    A depth row belongs to the contents when at least ``row_coverage`` of the run width is
    content there; the block of such rows that contains the opening is followed back until it
    breaks off for good.  Sparse noise behind the back wall never reaches that coverage.
    """
    hit, depths = _hit_grid(front, side, run, content, grid, upm, settings.max_depth_m * upm, settings.front_margin_m * upm)
    if hit.shape[0] == 0:
        return None
    rows = hit.mean(axis=0) >= settings.row_coverage
    if not rows.any():
        return None
    gap = max(1, int(round(settings.row_gap_m * upm / grid.pixel_size)))
    first = int(np.argmax(rows & (depths >= -settings.front_margin_m * upm)))
    last = first
    for index in range(first, len(rows)):
        if rows[index]:
            last = index
        elif index - last > gap:
            break
    coverage = float(hit[:, first:last + 1].any(axis=1).mean())
    return float(depths[first]), float(depths[last]), coverage


def _frame(front: _Chain, side: float, point: Iterable[float]) -> tuple[float, float]:
    vector = np.asarray(tuple(point), dtype=float)
    return float(front.unit @ vector), side * (float(front.normal @ vector) - front.offset)


def _find_panel(segments: list[_Segment], front: _Chain, side: float, end: float, upm: float, settings: NicheSettings) -> tuple[float, float, bool] | None:
    """A perpendicular wall / room edge that starts at the front line near ``end`` and runs back into the recess.

    Returns (along of the panel, across of its far end).  A wall that also runs forward
    beyond the front line is a through wall, not a side panel.
    """
    best: tuple[float, float, float, bool] | None = None
    forward = settings.panel_forward_m * upm
    for segment in segments:
        if _angle_gap(segment.angle, front.angle) < math.radians(90.0 - 8.0):
            continue
        ends = [_frame(front, side, point) for point in (segment.line.coords[0], segment.line.coords[-1])]
        ends.sort(key=lambda item: abs(item[1]))
        (near_along, near_across), (far_along, far_across) = ends
        if min(near_across, far_across) < -forward or abs(near_across) > forward:
            continue
        if far_across < settings.panel_min_length_m * upm:
            continue
        along = (near_along + far_along) / 2.0
        gap = abs(along - end)
        if gap > settings.panel_reach_m * upm:
            continue
        measured = segment.source == "wall"
        if best is None or (measured, -gap) > (best[3], -best[0]):
            best = (gap, along, far_across, measured)
    return None if best is None else (best[1], best[2], best[3])


def _side_chain_back(front: _Chain, side: float, chains: list[_Chain], span: tuple[float, float], lower: float, upper: float, upm: float, settings: NicheSettings) -> tuple[float, str] | None:
    limit = math.radians(4.0)
    best: tuple[float, str] | None = None
    for other in chains:
        if other is front or _angle_gap(other.angle, front.angle) > limit:
            continue
        pts = [np.asarray(point) for member in other.members for point in (member.line.coords[0], member.line.coords[-1])]
        depth = side * (float(np.mean([front.normal @ point for point in pts])) - front.offset)
        support = sum(member.line.length for member in other.members)
        if not lower <= depth <= upper or support < settings.chain_min_support_m * upm:
            continue
        if _extent_gap(front, other, span) > settings.chain_adjacency_m * upm:
            continue
        source = "wall_line" if any(member.source == "wall" for member in other.members) else "room_edge"
        if best is None or (source == "wall_line" and best[1] != "wall_line") or (source == best[1] and depth < best[0]):
            best = (depth, source)
    return best


def _wall_point_back(points: np.ndarray | None, front: _Chain, side: float, span: tuple[float, float], lower: float, upper: float, upm: float, settings: NicheSettings) -> tuple[float, int] | None:
    """Nearest dense line of wall points parallel to the front inside [lower, upper]."""
    if points is None:
        return None
    xy = points[:, :2]
    along = xy @ front.unit
    across = side * (xy @ front.normal - front.offset)
    margin = settings.wall_point_end_margin_m * upm
    selected = (along >= span[0] + margin) & (along <= span[1] - margin) & (across >= lower) & (across <= upper)
    if int(selected.sum()) < settings.wall_point_min_count:
        return None
    along_s, across_s = along[selected], across[selected]
    bin_size = settings.wall_point_bin_m * upm
    index = np.floor((across_s - lower) / bin_size).astype(int)
    counts = np.bincount(index).astype(float)
    smooth = np.convolve(counts, np.ones(3), mode="same")[: len(counts)]
    for position in range(len(smooth)):
        if smooth[position] < settings.wall_point_min_count:
            continue
        if position > 0 and smooth[position - 1] > smooth[position]:
            continue
        if position + 1 < len(smooth) and smooth[position + 1] > smooth[position]:
            continue
        members = (index >= position - 1) & (index <= position + 1)
        if not members.any():
            continue
        if float(np.ptp(along_s[members])) < settings.wall_point_min_extent_m * upm:
            continue
        return float(np.median(across_s[members])), int(members.sum())
    return None


def _plan_niche(
    front: _Chain, side: float, run: tuple[float, float], chains: list[_Chain], segments: list[_Segment], content: np.ndarray, free: np.ndarray,
    points: np.ndarray | None, grid: RasterGrid, upm: float, settings: NicheSettings,
) -> tuple[tuple[float, float], float, str, dict[str, Any]] | None:
    reach = settings.max_depth_m * upm
    extent = _contents_extent(front, side, run, content, grid, upm, settings)
    if extent is None:
        return None
    _, contents_far, coverage = extent
    if coverage < settings.min_run_coverage:
        return None
    if contents_far < settings.min_depth_m * upm:
        return None
    tolerance = settings.evidence_tolerance_m * upm
    start_panel = _find_panel(segments, front, side, run[0], upm, settings)
    stop_panel = _find_panel(segments, front, side, run[1], upm, settings)
    if settings.require_panel and start_panel is None and stop_panel is None:
        return None
    # A room-outline jog is not a side panel: at least one panel must be a measured wall.
    if settings.require_panel and not any(panel[2] for panel in (start_panel, stop_panel) if panel is not None):
        return None
    reach_end = settings.panel_reach_m * upm
    start = start_panel[0] if start_panel else (front.start if abs(run[0] - front.start) <= reach_end else run[0])
    stop = stop_panel[0] if stop_panel else (front.end if abs(run[1] - front.end) <= reach_end else run[1])
    span = (min(start, run[0]) if start_panel else start, max(stop, run[1]) if stop_panel else stop)
    width = (span[1] - span[0]) / upm
    if not settings.min_width_m <= width <= settings.max_width_m:
        return None
    lower = max(settings.min_depth_m * upm, contents_far - tolerance)
    upper = min(reach, contents_far + settings.back_slack_m * upm)
    info: dict[str, Any] = {"contents_far_m": contents_far / upm, "panels": int(start_panel is not None) + int(stop_panel is not None)}
    candidate = _side_chain_back(front, side, chains, span, lower, upper, upm, settings)
    wall_points = _wall_point_back(points, front, side, span, lower, upper, upm, settings)
    panel_ends = [panel[1] for panel in (start_panel, stop_panel) if panel is not None and lower <= panel[1] <= upper and panel[1] <= reach]
    if candidate is not None and candidate[1] == "wall_line":
        depth, evidence = candidate
    elif wall_points is not None:
        depth, evidence = wall_points[0], "wall_points"
        info["wall_points"] = wall_points[1]
    elif candidate is not None:
        depth, evidence = candidate
    elif panel_ends:
        depth, evidence = float(np.mean(panel_ends)), "side_panel"
    else:
        depth, evidence = contents_far, "contents"
    depth = min(max(depth, settings.min_depth_m * upm), reach)
    return span, depth, evidence, info


# -- applying ---------------------------------------------------------------

def _close_side_panels(layers: dict[str, list[dict[str, Any]]], front: _Chain, side: float, span: tuple[float, float], depth: float, upm: float) -> None:
    """Make sure a wall runs from the opening to the back line at both ends of the niche."""
    walls = layers["A-WALL"]
    for end in span:
        covered: list[tuple[float, float]] = []
        for entity in walls:
            pts = entity.get("points", [])
            if entity.get("type") != "polyline" or len(pts) != 2 or entity.get("closed"):
                continue
            if _angle_gap(_angle_of(LineString(pts)), front.angle) < math.radians(90.0 - 8.0):
                continue
            (s0, t0), (s1, t1) = _frame(front, side, pts[0]), _frame(front, side, pts[1])
            if abs(s0 - end) > 0.10 * upm or abs(s1 - end) > 0.10 * upm:
                continue
            low, high = max(0.0, min(t0, t1)), min(depth, max(t0, t1))
            if high > low:
                covered.append((low, high))
        cursor = 0.0
        missing: list[tuple[float, float]] = []
        for low, high in sorted(covered):
            if low - cursor > 0.05 * upm:
                missing.append((cursor, low))
            cursor = max(cursor, high)
        if depth - cursor > 0.05 * upm:
            missing.append((cursor, depth))
        for low, high in missing:
            first, second = (list(map(float, _frame_point(front, side, end, value)[0])) for value in (low, high))
            walls.append({"type": "polyline", "points": [first, second], "closed": False, "_rms": 0.0, "_max_residual": 0.0, "_axis_snapped": True, "_source": "niche_panel"})


def _clear_beyond(layers: dict[str, list[dict[str, Any]]], front: _Chain, side: float, span: tuple[float, float], depth: float, upm: float, settings: NicheSettings) -> int:
    """Drop everything that lies behind the back wall (furniture, clutter, stray wall pieces)."""
    inset = settings.beyond_inset_m * upm
    along_range = (span[0] + inset, span[1] - inset)
    across_range = (depth + inset, depth + settings.beyond_reach_m * upm)

    zone = Polygon([tuple(point) for point in _frame_point(front, side, [span[0], span[1], span[1], span[0]], [depth, depth, across_range[1], across_range[1]])])

    def beyond(point: Iterable[float]) -> bool:
        along, across = _frame(front, side, point)
        return along_range[0] <= along <= along_range[1] and across_range[0] <= across <= across_range[1]

    def inside(entity: dict[str, Any]) -> bool:
        pts = entity.get("points") or ([entity["point"]] if entity.get("point") is not None else [])
        if len(pts) == 0:
            return False
        if entity.get("type") == "text" or len(pts) == 1:
            return beyond(pts[0])
        if entity.get("closed") and len(pts) >= 3:
            polygon = Polygon(pts).buffer(0)
            # Mostly behind the back wall (a blob straddling the line counts, its front part is noise too).
            return not polygon.is_empty and polygon.intersection(zone).area >= 0.5 * polygon.area
        return all(beyond(point) for point in pts)

    removed = 0
    for name in ("A-FURN", "A-FURN-DETL", "A-EINB", "A-WALL", "A-ANNO"):
        kept: list[dict[str, Any]] = []
        for entity in layers.get(name, []):
            drop = False
            if name == "A-ANNO":
                drop = entity.get("type") == "text" and " h=" in str(entity.get("text", "")) and inside(entity)
            elif name == "A-EINB":
                owner = entity.get("_einb_owner")
                niche_owned = entity.get("_einb_kind") == "niche" or (owner is not None and owner >= NICHE_ID_BASE)
                drop = not niche_owned and inside(entity)
            else:
                drop = inside(entity)
            if drop:
                removed += 1
            else:
                kept.append(entity)
        layers[name] = kept
    return removed
