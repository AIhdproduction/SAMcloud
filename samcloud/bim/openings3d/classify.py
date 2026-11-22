"""Door / passage / no-opening classification, open-leaf assignment and exterior door scan.

Runs after the gap detection of :mod:`.detect`.  Every door or passage candidate gets its frame, head and
headroom measured (:mod:`.frames`) and is classified by plain evidence rules (thresholds in
``config/bim_openings.yaml`` section ``classify``):

* **door**: frame evidence (a jamb strip and the head strip of door-class points), or a closed leaf filling the
  opening, or a lintel (headroom above the opening, measured head high enough) on an observed wall.
* **passage**: no door evidence, the opening reaches (nearly) the ceiling (headroom below
  ``passage_max_headroom_m``) and its cells are measured free of points.
* **none**: neither door evidence nor a measured free gap (or the opening is not on an observed wall): the
  candidate is dropped, the wall closes.  Weak door evidence (a single jamb or head strip, door-class cells in the
  gap) keeps the safe default, a door.

An open leaf never makes a door; it is only assigned (hinge, swing, angle) to a door whose own jamb carries the
hinge (:mod:`.leaves`).  Wall lines are also scanned for door frames where no opening exists (entrances).
"""

from __future__ import annotations

import math
from dataclasses import replace
from typing import Any

import numpy as np

from samcloud.bim.plan_model import BuildingPlan

from .config import Settings
from .frames import FrameFeatures, measure_opening
from .leaves import Leaf, find_leaf
from .model import RefinedOpening
from .points import PointEvidence
from .walls import WallLine, opening_placement, storey_height_m


def classify_features(features: FrameFeatures, config: Settings, seeded: bool = True) -> tuple[str, str]:
    """(kind, reason) of one measured opening: ``door`` | ``passage`` | ``none``.

    ``seeded`` tells whether the 2D plan saw a door or passage there; only then does weak evidence (a single casing strip, door-class
    cells without a casing) keep the safe default, a door.  A gap that only the wall-plane measurement found needs real evidence.
    """
    cfg = config.classify
    frame = bool(features.frame_jambs) and features.head_frame
    # A closed leaf fills the opening with door-class points AND sits under a head strip; door-class points without a head (an open leaf
    # of the neighbouring door lying flat beside / in the gap) are no door.
    closed_leaf = features.door_fraction >= float(config.frames.closed_leaf_fraction) and features.head_cov >= float(cfg.closed_leaf_min_head_cov)
    on_wall = max(features.side_wall_cov.values(), default=0.0) >= float(cfg.min_side_wall_cov)
    # The head is a door head when it reaches the absolute door-head height OR a share of the storey height: the share is
    # independent of the metric scale of the plan (an approximate storey scale shrinks every height by the same factor).
    head_fraction = float(getattr(cfg, "lintel_min_head_fraction", 0.0) or 0.0)
    head_is_door_head = features.head_m is not None and (
        features.head_m >= float(cfg.lintel_min_head_m)
        or (head_fraction > 0.0 and features.ceiling_m > 0.0 and features.head_m >= head_fraction * features.ceiling_m)
    )
    # A lintel over a gap that only the wall-plane measurement found needs that gap measured free (furniture or an unscanned
    # stretch in front of the wall leaves the same "head" signature); a gap the plan saw keeps the lenient rule.
    # Such a gap must also show the floor running through the wall band (furniture against the wall hides wall AND floor; its
    # unmeasured cells look "free" without photographs).
    floor_through = seeded or features.floor_fraction >= float(getattr(cfg, "unseeded_min_floor_fraction", 0.0) or 0.0)
    lintel_free = seeded or (features.free_fraction >= float(getattr(cfg, "lintel_unseeded_min_free_fraction", 0.0) or 0.0) and floor_through)
    lintel = (
        head_is_door_head and features.headroom_m is not None
        and features.headroom_m >= float(cfg.lintel_min_headroom_m) and lintel_free
    )
    weak = bool(features.frame_jambs or features.head_frame or features.flat_leaf_sides) or features.door_fraction >= float(cfg.weak_door_fraction)
    # Headroom below the limit, or no point at all between the search height and the ceiling (the gap runs up to the ceiling).
    to_ceiling = (features.head_m is None and bool(cfg.passage_unmeasured_head_open)) or (features.headroom_m is not None and features.headroom_m < float(cfg.passage_max_headroom_m))
    if frame:
        return "door", f"frame ({'+'.join(features.frame_jambs)} jamb + head strip)"
    if closed_leaf:
        return "door", f"closed leaf (door cells {features.door_fraction:.2f}, head strip {features.head_cov:.2f})"
    if lintel and on_wall:
        return "door", f"lintel (headroom {features.headroom_m:.2f} m, head {features.head_m:.2f} m)"
    if on_wall and to_ceiling and features.free_fraction >= float(cfg.passage_min_free_fraction) and floor_through:
        shown = "no point up to the ceiling" if features.headroom_m is None else f"headroom {features.headroom_m:.2f} m"
        return "passage", f"free gap to the ceiling ({shown}, free cells {features.free_fraction:.2f}), no frame"
    if weak and seeded:
        return "door", "weak door evidence kept (safe default)"
    if not on_wall:
        return "none", f"no door evidence and not on an observed wall (wall beside the opening {max(features.side_wall_cov.values(), default=0.0):.2f})"
    return "none", f"no door evidence and no measured free gap (headroom {features.headroom_m}, free cells {features.free_fraction:.2f}, blocked {features.blocked_fraction:.2f}, weak evidence without a plan door ignored: {weak and not seeded})"


def _mirrored(line: WallLine, angle: float) -> bool:
    return float(np.dot(np.array([math.cos(angle), math.sin(angle)]), line.d)) < 0.0


def _axis_sides(names: list[str], mirrored: bool) -> list[str]:
    flip = {"left": "right", "right": "left"}
    return sorted(flip[name] if mirrored else name for name in names)


def _describe(features: FrameFeatures) -> str:
    head = "-" if features.head_m is None else f"{features.head_m:.2f}"
    room = "-" if features.headroom_m is None else f"{features.headroom_m:.2f}"
    return (
        f"head {head} m, headroom {room} m, jamb cov L{features.jamb_cov.get('left', 0):.2f}/R{features.jamb_cov.get('right', 0):.2f}, head strip {features.head_cov:.2f}"
        f" (peak {features.head_peak_m}), door cells {features.door_fraction:.2f}, free {features.free_fraction:.2f}, blocked {features.blocked_fraction:.2f}, side wall L{features.side_wall_cov.get('left', 0):.2f}/R{features.side_wall_cov.get('right', 0):.2f}"
    )


def _apply_features(record: RefinedOpening, kind: str, reason: str, features: FrameFeatures, line: WallLine, config: Settings) -> RefinedOpening:
    mirrored = _mirrored(line, record.axis_angle)
    head_m, source = record.head_m, record.height_source
    closed = features.door_fraction >= float(config.frames.closed_leaf_fraction)
    if features.head_m is not None and not closed and float(config.classify.min_head_m) <= features.head_m <= features.ceiling_m:
        head_m, source = round(features.head_m, 3), "points"
    return replace(
        record, kind=kind, head_m=head_m, height_source=source, headroom_m=None if features.headroom_m is None else round(features.headroom_m, 3),
        frame_jambs=_axis_sides(features.frame_jambs, mirrored), head_frame=features.head_frame, classification=kind,
        evidence=f"{record.evidence}; {kind}: {reason}; {_describe(features)}"[:600],
    )


def _same_leaf(first: Leaf, second: Leaf, config: Settings) -> bool:
    cfg = config.leaf
    if math.dist(first.hinge, second.hinge) > float(cfg.same_hinge_m):
        return False
    a = np.asarray(first.free) - np.asarray(first.hinge)
    b = np.asarray(second.free) - np.asarray(second.hinge)
    cosine = float(a @ b / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-9))
    return math.degrees(math.acos(float(np.clip(cosine, -1.0, 1.0)))) < float(cfg.same_direction_deg)


def assign_leaves(records: list[RefinedOpening], door_points: np.ndarray | None, config: Settings, units_per_metre: float = 1.0) -> list[RefinedOpening]:
    """Fill hinge / swing / angle for doors that own an open leaf; passages and windows never get a leaf.

    The leaf must hinge at the door's own jamb (the search is restricted to the jamb).  When the neighbouring door of a
    close pair finds the same physical leaf, only the door whose jamb the hinge is closest to keeps it.
    """
    if door_points is None or not len(door_points):
        return records
    found: dict[int, Leaf] = {}
    for index, record in enumerate(records):
        if record.kind != "door":
            continue
        leaf = find_leaf(door_points, (float(record.center[0]) / units_per_metre, float(record.center[1]) / units_per_metre), float(record.axis_angle), float(record.width_m), config)
        if leaf is not None:
            found[index] = leaf
    for index in sorted(found):
        for other in sorted(found):
            if other <= index or index not in found or other not in found or not _same_leaf(found[index], found[other], config):
                continue
            loser = other if found[index].hinge_offset_m <= found[other].hinge_offset_m else index
            del found[loser]
    result = list(records)
    for index, leaf in found.items():
        result[index] = replace(
            records[index], hinge_side=leaf.hinge_side, swing_side=leaf.swing_side, leaf_angle_deg=round(leaf.angle_deg, 1), leaf_detected=True, leaf=leaf.to_dict(),
            evidence=f"{records[index].evidence}; leaf hinge {leaf.hinge_side}, opens {leaf.angle_deg:.0f} deg to {'+' if leaf.swing_side > 0 else '-'}normal"[:600],
        )
    return result


def _segment_distance(point: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    along = b - a
    norm2 = float(along @ along)
    t = 0.0 if norm2 <= 0.0 else float(np.clip((point - a) @ along / norm2, 0.0, 1.0))
    return float(np.linalg.norm(point - (a + along * t)))


def drop_leaf_shadows(records: list[RefinedOpening], config: Settings, units_per_metre: float) -> tuple[list[RefinedOpening], list[dict[str, Any]]]:
    """Remove door candidates that are only the open leaf of another door lying along / inside their wall.

    A leaf that swung open flat beside a wall fills the wall plane with door-class points and looks like a closed leaf in a gap
    (a door in the wall where the leaf lies).  The FRAME decides where a door is: a candidate without any casing strip (no jamb strip,
    no head strip) whose centre lies within ``classify.leaf_shadow_distance_m`` of the measured leaf of a door on ANOTHER wall line is
    that leaf and no opening.  A candidate with frame evidence is never dropped here.
    """
    reach = float(config.classify.leaf_shadow_distance_m)
    dropped: list[dict[str, Any]] = []
    kept: list[RefinedOpening] = []
    for record in records:
        framed = bool(record.frame_jambs) or record.head_frame
        shadow_of = None
        if record.kind in ("door", "passage") and not framed and not record.leaf_detected:
            centre = np.asarray(record.center, float) / units_per_metre
            for other in records:
                if other is record or not other.leaf_detected or not other.leaf or other.line == record.line:
                    continue
                if _segment_distance(centre, np.asarray(other.leaf["hinge"], float), np.asarray(other.leaf["free"], float)) <= reach:
                    shadow_of = other
                    break
        if shadow_of is None:
            kept.append(record)
            continue
        dropped.append({
            "center": record.center, "width_m": record.width_m, "axis_angle": record.axis_angle, "wall_thickness_m": record.wall_thickness_m, "host_wall_id": record.host_wall_id,
            "line": record.line, "u": [record.u0, record.u1], "plan_seed": record.plan_seed, "features": {}, "was": record.kind,
            "reason": f"no frame evidence and within {reach:.2f} m of the open leaf of the door on {shadow_of.line} (hinge {shadow_of.leaf['hinge']}): this is that leaf, not an opening",
        })
    return kept, dropped


# --------------------------------------------------------------------------
# exterior / entrance door scan
# --------------------------------------------------------------------------

def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    index = np.flatnonzero(mask)
    if index.size == 0:
        return []
    breaks = np.flatnonzero(np.diff(index) > 1)
    return list(zip(np.r_[index[0], index[breaks + 1]].tolist(), np.r_[index[breaks], index[-1]].tolist()))


def scan_frames(line: WallLine, grids: dict[str, np.ndarray], ceiling_m: float, config: Settings) -> list[dict[str, Any]]:
    """Door frames (jamb strip + head strip) on a wall line, as candidates (u0, u1, features), best first.

    A closed leaf is one wide run of door-class columns, an empty frame is two narrow runs (the jambs); both give
    the candidate opening between the runs.  Only candidates with a measured casing (``classify`` frame rule) count.
    """
    cfg = config.scan
    cell = float(config.grid.cell_m)
    door = grids["door"]
    nv, nu = door.shape
    rows = slice(int(round(float(cfg.col_rows_m[0]) / cell)), min(int(round(float(cfg.col_rows_m[1]) / cell)), nv))
    cover = (door[rows] >= int(cfg.col_min_points)).mean(0)
    tall = cover >= float(cfg.col_coverage)
    runs = [(a, b) for a, b in _runs(np.convolve(tall.astype(float), np.ones(3), "same") >= 1.0)]
    candidates: list[tuple[float, float]] = []
    for a, b in runs:
        width = (b - a + 1) * cell
        if float(cfg.min_width_m) + 2 * float(cfg.run_shrink_m) <= width <= float(cfg.max_width_m) + 2 * float(cfg.run_shrink_m):
            candidates.append((a * cell + float(cfg.run_shrink_m), (b + 1) * cell - float(cfg.run_shrink_m)))
    for (a0, b0), (a1, b1) in zip(runs, runs[1:]):
        if (b0 - a0 + 1) * cell > float(cfg.max_jamb_run_m) or (b1 - a1 + 1) * cell > float(cfg.max_jamb_run_m):
            continue
        gap = (a1 - b0 - 1) * cell
        if float(cfg.min_width_m) <= gap <= float(cfg.max_width_m):
            candidates.append(((b0 + 1) * cell, a1 * cell))
    found: list[dict[str, Any]] = []
    for u0, u1 in candidates:
        if u0 < float(cfg.end_margin_m) or u1 > line.length_m - float(cfg.end_margin_m):
            continue
        features = measure_opening(grids, u0, u1, ceiling_m, config)
        if features.frame_jambs and features.head_frame and not features.flat_leaf_sides:
            found.append({"u0": u0, "u1": u1, "features": features})
    found.sort(key=lambda item: (-len(item["features"].frame_jambs), -item["features"].head_cov, item["u0"]))
    return found


def _overlaps(line: WallLine, u0: float, u1: float, records: list[RefinedOpening], margin: float) -> bool:
    return any(record.line == line.id and min(record.u1, u1) - max(record.u0, u0) > -margin for record in records)


def _plan_opening_there(line: WallLine, u0: float, u1: float, plan: BuildingPlan, config: Settings) -> bool:
    """True when an opening of the plan (door, passage or window; also one the refinement has not measured) stands in the span."""
    upm = plan.units_per_metre
    margin, reach = float(config.scan.overlap_margin_m), float(config.scan.plan_opening_reach_m)
    for opening in plan.openings:
        relative = np.asarray(opening.center, float) - line.a
        along, across = float(relative @ line.d) / upm, abs(float(relative @ line.n)) / upm
        if across < float(opening.wall_thickness_m) / 2.0 + reach and u0 - margin <= along <= u1 + margin:
            return True
    return False


def scan_entrances(lines: list[WallLine], evidence_points: PointEvidence, plan: BuildingPlan, records: list[RefinedOpening], config: Settings) -> tuple[list[RefinedOpening], list[dict[str, Any]]]:
    """Add a door for every measured door frame on a wall line that has no opening there yet (never overlaps one)."""
    ceiling = storey_height_m(plan)
    added: list[RefinedOpening] = []
    report: list[dict[str, Any]] = []
    margin = float(config.scan.overlap_margin_m)
    for line in lines:
        if "any" not in evidence_points.grids.get(line.id, {}):
            continue
        for candidate in scan_frames(line, evidence_points.grids[line.id], ceiling, config):
            u0, u1 = candidate["u0"], candidate["u1"]
            features: FrameFeatures = candidate["features"]
            item = {"line": line.id, "u": [round(u0, 2), round(u1, 2)], "features": features.to_dict()}
            if _overlaps(line, u0, u1, [*records, *added], margin) or _plan_opening_there(line, u0, u1, plan, config):
                item["result"] = "skipped: an opening exists there"
                report.append(item)
                continue
            kind, reason = classify_features(features, config)
            if kind != "door":
                item["result"] = f"rejected: {kind} ({reason})"
                report.append(item)
                continue
            centre, host, angle, thickness = opening_placement(line, (u0 + u1) / 2, plan)
            base = RefinedOpening(
                id="", kind="door", center=centre, width_m=round(u1 - u0, 3), axis_angle=round(angle, 4), wall_thickness_m=round(thickness, 3), host_wall_id=host, sill_m=0.0,
                head_m=float(config.doors.default_head_m), height_source="default", evidence="door frame found by the wall scan (no opening there before)",
                confidence=float(config.scan.confidence), line=line.id, u0=round(u0, 3), u1=round(u1, 3),
            )
            added.append(_apply_features(base, kind, reason, features, line, config))
            item["result"] = "added"
            report.append(item)
    return added, report


# --------------------------------------------------------------------------
# entry point
# --------------------------------------------------------------------------

_STRENGTH = {"frame": 4, "closed leaf": 3, "lintel": 3, "passage": 2, "weak": 1}


def _strength(kind: str, reason: str, features: FrameFeatures) -> tuple[int, int, float]:
    """Evidence rank of a classified opening: casing > leaf / lintel > passage > weak; ties by jamb count and head strip."""
    label = "passage" if kind == "passage" else next((name for name in ("frame", "closed leaf", "lintel", "weak") if reason.startswith(name)), "weak")
    return _STRENGTH[label], len(features.frame_jambs), features.head_cov


def _dropped_item(record: RefinedOpening, line: WallLine, reason: str, features: FrameFeatures) -> dict[str, Any]:
    return {
        "center": record.center, "width_m": record.width_m, "axis_angle": record.axis_angle, "wall_thickness_m": record.wall_thickness_m, "host_wall_id": record.host_wall_id,
        "line": line.id, "u": [record.u0, record.u1], "plan_seed": record.plan_seed, "reason": reason, "features": features.to_dict(), "was": record.kind,
    }


def resolve_overlaps(
    measured: list[tuple[RefinedOpening, WallLine, FrameFeatures, str, str]], evidence_points: PointEvidence, plan: BuildingPlan, config: Settings,
) -> tuple[list[RefinedOpening], list[dict[str, Any]]]:
    """Openings of one wall line never overlap and never share a casing: the weaker one is trimmed at the edge of the stronger one.

    Two door / passage candidates that overlap by more than ``classify.min_overlap_m`` were seeded by neighbouring plan openings
    (a door beside a passage in one wall gap).  The one with less evidence (:func:`_strength`) keeps only the stretch the other does
    not cover and is measured again there with the shared edge excluded (that jamb strip is the other opening's casing); it is
    dropped when it is narrower than ``doors.min_width_m`` afterwards or no longer a door / passage.
    """
    ceiling = storey_height_m(plan)
    margin = float(config.classify.min_overlap_m)
    minimum = float(config.doors.min_width_m)
    items: list[list[Any] | None] = [list(item) for item in measured]
    dropped: list[dict[str, Any]] = []
    changed = True
    while changed:
        changed = False
        live = sorted((index for index, item in enumerate(items) if item is not None), key=lambda index: (items[index][1].id, items[index][0].u0, index))  # type: ignore[index]
        for position, first in enumerate(live):
            if changed:
                break
            for second in live[position + 1:]:
                a, b = items[first], items[second]
                if a is None or b is None or a[1].id != b[1].id:
                    continue
                if min(a[0].u1, b[0].u1) - max(a[0].u0, b[0].u0) <= margin:
                    continue
                loser_index, winner_index = (first, second) if _strength(a[3], a[4], a[2]) < _strength(b[3], b[4], b[2]) else (second, first)
                loser, winner = items[loser_index], items[winner_index]
                assert loser is not None and winner is not None
                record, line = loser[0], loser[1]
                if record.u0 < winner[0].u0:
                    u0, u1, shared = record.u0, winner[0].u0, "right"
                else:
                    u0, u1, shared = winner[0].u1, record.u1, "left"
                changed = True
                if u1 - u0 < minimum:
                    dropped.append(_dropped_item(record, line, f"overlaps the {winner[0].kind} beside it on {line.id} and is too narrow after trimming", loser[2]))
                    items[loser_index] = None
                    break
                features = measure_opening(evidence_points.grids[line.id], u0, u1, ceiling, config, shared_edges=(shared,))
                kind, reason = classify_features(features, config, seeded=record.plan_seed is not None)
                if kind == "none":
                    dropped.append(_dropped_item(record, line, f"overlaps the {winner[0].kind} beside it on {line.id}; trimmed stretch: {reason}", features))
                    items[loser_index] = None
                    break
                centre, host, angle, thickness = opening_placement(line, (u0 + u1) / 2.0, plan, record.host_wall_id)
                trimmed = replace(
                    record, u0=round(u0, 3), u1=round(u1, 3), width_m=round(u1 - u0, 3), center=centre, host_wall_id=host, axis_angle=round(angle, 4),
                    wall_thickness_m=round(thickness, 3), evidence=f"{record.evidence}; trimmed at the {winner[0].kind} beside it (shared casing)"[:600],
                )
                items[loser_index] = [_apply_features(trimmed, kind, reason, features, line, config), line, features, kind, reason]
                break
    return [item[0] for item in items if item is not None], dropped


def restore_framed_duplicates(
    kept: list[RefinedOpening], duplicates: list[RefinedOpening], by_line: dict[str, WallLine], evidence_points: PointEvidence, plan: BuildingPlan, config: Settings,
) -> list[RefinedOpening]:
    """A door / passage candidate with frame evidence (jamb strip AND head strip) is never lost to the duplicate filter.

    The duplicate filter ranks candidates by confidence before anything is measured; a window rectangle or a leaf-made candidate can
    outrank a real doorway.  A rejected duplicate that is measured as a framed door and whose span no kept opening of its wall line covers
    (more than ``classify.min_overlap_m``) is restored (at worst without a measured leaf).
    """
    ceiling = storey_height_m(plan)
    margin = float(config.classify.min_overlap_m)
    restored: list[RefinedOpening] = []
    for record in sorted(duplicates, key=lambda item: (item.line or "", item.u0)):
        line = by_line.get(record.line or "")
        if record.kind not in ("door", "passage") or line is None or "any" not in evidence_points.grids.get(line.id, {}):
            continue
        features = measure_opening(evidence_points.grids[line.id], record.u0, record.u1, ceiling, config)
        if not (features.frame_jambs and features.head_frame):
            continue
        if any(other.line == record.line and min(other.u1, record.u1) - max(other.u0, record.u0) > margin for other in [*kept, *restored]):
            continue
        kind, reason = classify_features(features, config, seeded=record.plan_seed is not None)
        if kind == "door":
            restored.append(_apply_features(record, kind, f"{reason}; restored after the duplicate filter (frame evidence)", features, line, config))
    return restored


def classify_openings(
    openings: list[RefinedOpening], lines: list[WallLine], evidence_points: PointEvidence, plan: BuildingPlan, config: Settings, duplicates: list[RefinedOpening] | None = None,
) -> tuple[list[RefinedOpening], list[dict[str, Any]], dict[str, Any]]:
    """(kept openings, dropped candidates with their measurements, summary) after classification and the entrance scan."""
    by_line = {line.id: line for line in lines}
    ceiling = storey_height_m(plan)
    kept: list[RefinedOpening] = []
    dropped: list[dict[str, Any]] = []
    measured: list[tuple[RefinedOpening, WallLine, FrameFeatures, str, str]] = []
    for record in openings:
        line = by_line.get(record.line or "")
        if record.kind not in ("door", "passage") or line is None or "any" not in evidence_points.grids.get(line.id, {}):
            kept.append(record)
            continue
        features = measure_opening(evidence_points.grids[line.id], record.u0, record.u1, ceiling, config)
        kind, reason = classify_features(features, config, seeded=record.plan_seed is not None)
        if kind == "none":
            dropped.append(_dropped_item(record, line, reason, features))
            continue
        measured.append((_apply_features(record, kind, reason, features, line, config), line, features, kind, reason))
    resolved, trimmed_out = resolve_overlaps(measured, evidence_points, plan, config)
    dropped.extend(trimmed_out)
    kept.extend(resolved)
    kept.extend(restore_framed_duplicates(kept, duplicates or [], by_line, evidence_points, plan, config))
    added, scan_report = scan_entrances(lines, evidence_points, plan, kept, config)
    kept.extend(added)
    kept = assign_leaves(kept, evidence_points.door_points, config, plan.units_per_metre)
    kept, shadows = drop_leaf_shadows(kept, config, plan.units_per_metre)
    dropped.extend(shadows)
    ordered = sorted(kept, key=lambda item: (item.center[1], item.center[0]))
    for index, record in enumerate(ordered, start=1):
        record.id = f"C{index:03d}"
    summary = {
        "doors": sum(item.kind == "door" for item in ordered), "passages": sum(item.kind == "passage" for item in ordered), "dropped": len(dropped),
        "added_by_scan": len(added), "leaves": sum(bool(item.leaf_detected) for item in ordered), "scan": scan_report,
    }
    return ordered, dropped, summary
