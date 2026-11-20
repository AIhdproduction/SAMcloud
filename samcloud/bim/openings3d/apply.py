"""Hook for the clean plan: replace the 2D-detected openings by the refined ones.

``apply_refined_openings`` is the only entry the indoor plan generator calls.  It uses
``openings_3d.json`` only when its wall fingerprint matches the wall set of the level's pass-1 plan
(``plan_model_pass1.json``, else ``plan_model.json``) (so refined openings of a different wall set are never applied) and leaves the 2D
openings untouched otherwise.  The returned info dict is written into ``plan_model.json`` (``source``)
so run_all can tell whether the plan already contains the current refinement.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any

import numpy as np
from shapely.geometry import LineString

from samcloud.bim.invariants import door_backing
from samcloud.bim.plan_model import load_refine_plan
from samcloud.ortho.vector.indoor import Opening

from .config import Settings, load_openings_config
from .model import OpeningsResult, RefinedOpening, load_openings_3d, openings_3d_path, openings_3d_sha256
from .walls import walls_fingerprint, walls_match, wall_lines

_LOGGER = logging.getLogger(__name__)
SNAP_TOLERANCE_M = 0.30  # a refined opening must lie this close to a measured wall face line of the current pass
PARALLEL_TOLERANCE_DEG = 5.0


def load_matching_openings(cache_directory: Path | str, config: Settings | None = None) -> OpeningsResult | None:
    """``openings_3d.json`` of the level when its wall fingerprint matches the level's plan_model.json, else None."""
    directory = Path(cache_directory)
    if not openings_3d_path(directory).is_file():
        return None
    try:
        settings = config or load_openings_config()
        result = load_openings_3d(directory)
        plan = load_refine_plan(directory)
        lines = wall_lines(plan, settings)
        tolerance = settings.freshness.wall_tolerance_m * plan.units_per_metre
        if not walls_match(list(result.inputs.get("walls", [])), walls_fingerprint(lines), tolerance):
            _LOGGER.info("openings3d: openings_3d.json belongs to a different wall set; keeping the 2D openings")
            return None
        return result
    except Exception:  # noqa: BLE001 - unreadable / missing plan model: no refinement
        _LOGGER.warning("openings3d: could not read the refined openings; keeping the 2D openings", exc_info=True)
        return None


def _snap_to_face(centre: np.ndarray, angle: float, face_lines: list[LineString], tolerance: float) -> tuple[np.ndarray, float] | None:
    """Project the centre onto the nearest roughly parallel face line within ``tolerance``; None when no face is near."""
    original_axis = np.asarray((math.cos(angle), math.sin(angle)))
    best: tuple[float, np.ndarray, np.ndarray] | None = None
    for line in face_lines:
        start, end = np.asarray(line.coords[0], float), np.asarray(line.coords[-1], float)
        axis = end - start
        length = float(np.linalg.norm(axis))
        if length < 1e-9:
            continue
        axis = axis / length
        deviation = math.degrees(math.asin(float(np.clip(abs(axis[0] * original_axis[1] - axis[1] * original_axis[0]), 0.0, 1.0))))
        if deviation > PARALLEL_TOLERANCE_DEG:
            continue
        offset = abs(float(np.dot(centre - start, np.asarray((-axis[1], axis[0])))))
        if offset > tolerance:
            continue
        projected = start + axis * float(np.dot(centre - start, axis))
        if best is None or offset < best[0]:
            best = (offset, projected, axis)
    if best is None:
        return None
    return best[1], math.atan2(float(best[2][1]), float(best[2][0])) % math.pi


class _ClosedSpan:
    """A dropped door candidate seen as an interval on its wall line (``line``, ``u0``, ``u1``)."""

    def __init__(self, item: dict[str, Any]) -> None:
        self.line = item.get("line")
        self.u0, self.u1 = (item.get("u") or [0.0, 0.0])[:2]


def host_side_spans(openings: list[RefinedOpening], plan: Any, config: Settings, closed: list[Any] = ()) -> list[dict[str, Any]]:
    """Stretches of the ORIGINAL host wall (pass-1 wall pieces) beside each refined door or passage, up to ``classify.host_reach_m``.

    The wall stage continues the host wall over such a stretch only where the measured faces leave it open (a door that was
    measured as a gap in an unmeasured wall must not take the wall beside it away).  Centre, angle and width are the pass-1 wall
    piece's own, in plan units.
    """
    upm = plan.units_per_metre
    reach = float(config.classify.host_reach_m)
    lines = {line.id: line for line in wall_lines(plan, config)}
    by_id = {wall.id: wall for wall in plan.walls}
    spans: list[dict[str, Any]] = []
    margin = 0.05
    for opening in openings:
        line = lines.get(opening.line or "")
        if line is None or opening.kind not in ("door", "passage"):
            continue
        # every opening on the line (windows, other doors, passages, dropped candidates) keeps its own span free
        taken = sorted((other.u0 - margin, other.u1 + margin) for other in [*openings, *closed] if other is not opening and (other.line or "") == line.id)
        for lo, hi in ((opening.u0 - reach, opening.u0), (opening.u1, opening.u1 + reach)):
            for piece_id in line.pieces:
                a, b = (np.asarray(point, float) for point in by_id[piece_id].centreline)
                ua, ub = sorted((float((a - line.a) @ line.d) / upm, float((b - line.a) @ line.d) / upm))
                free = [(max(ua, lo), min(ub, hi))]
                for t0, t1 in taken:
                    free = [part for start, end in free for part in ((start, min(end, t0)), (max(start, t1), end))]
                for start, end in free:
                    if end - start < float(config.classify.host_min_stretch_m):
                        continue
                    middle = line.a + line.d * ((start + end) / 2.0) * upm
                    spans.append({
                        "center": [round(float(middle[0]), 4), round(float(middle[1]), 4)], "axis_angle": round(math.atan2(float(line.d[1]), float(line.d[0])) % math.pi, 5),
                        "width": round((end - start) * upm, 4), "thickness": round(float(by_id[piece_id].thickness_m) * upm, 4),
                    })
    return spans


def to_plan_opening(refined: RefinedOpening, units_per_metre: float, center: tuple[float, float], axis_angle: float) -> Opening:
    kind = {"door": "DOOR", "window": "WINDOW"}.get(refined.kind, "passage")
    return Opening(
        kind=kind, center=center, width=float(refined.width_m) * units_per_metre, axis_angle=axis_angle,
        wall_thickness=float(refined.wall_thickness_m) * units_per_metre, evidence=f"3d:{refined.height_source}; backing={door_backing(refined) or 'none'}; {refined.evidence}"[:200],
        class_evidence=float(refined.confidence), sill_m=float(refined.sill_m), head_m=float(refined.head_m), height_source=refined.height_source,
        mullion_u=tuple(float(value) for value in refined.mullion_u), glazed=bool(refined.glazed), confidence=float(refined.confidence),
        **_leaf_fields(refined, units_per_metre),
    )


def _leaf_fields(refined: RefinedOpening, units_per_metre: float = 1.0) -> dict[str, Any]:
    """Measured open leaf of a door (hinge / swing / angle); windows, passages and doors without a leaf carry none."""
    if refined.kind != "door" or not refined.leaf_detected:
        return {}  # no leaf measured: leaf_detected stays None, the 2D leaf detection of the clean plan still runs for this door
    fields: dict[str, Any] = {
        "hinge_side": refined.hinge_side, "swing_side": refined.swing_side, "leaf_angle_deg": refined.leaf_angle_deg, "leaf_detected": True, "leaf_mode": "open",
    }
    leaf = refined.leaf or {}
    if leaf.get("hinge") and leaf.get("free"):  # the measured leaf segment (plan units): the wall stage must never build a wall there
        fields["leaf_hinge"] = (float(leaf["hinge"][0]) * units_per_metre, float(leaf["hinge"][1]) * units_per_metre)
        fields["leaf_free"] = (float(leaf["free"][0]) * units_per_metre, float(leaf["free"][1]) * units_per_metre)
    return fields


def apply_refined_openings(
    openings: list[Opening],
    cache_directory: Path | str,
    face_lines: list[LineString],
    units_per_metre: float,
    config: Settings | None = None,
) -> tuple[list[Opening], dict[str, Any]]:
    """Replace ``openings`` by the refined ones when a matching ``openings_3d.json`` exists.

    Returns ``(openings, info)``. ``info`` is {} without an ``openings_3d.json``; otherwise it carries the file's
    ``sha256`` and ``applied`` (False when the fingerprint did not match, then ``openings`` is returned unchanged);
    on success also ``mode``, ``count`` and ``dropped`` (refined openings without a wall face nearby).
    """
    sha = openings_3d_sha256(cache_directory)
    if sha is None:
        return openings, {}
    result = load_matching_openings(cache_directory, config)
    if result is None:
        return openings, {"sha256": sha, "applied": False}
    snap = SNAP_TOLERANCE_M * units_per_metre
    replaced: list[Opening] = []
    dropped: list[str] = []
    for refined in result.openings:
        placed = _snap_to_face(np.asarray(refined.center, float), float(refined.axis_angle), face_lines, snap)
        if placed is None:
            dropped.append(refined.id)
            continue
        centre, angle = placed
        replaced.append(to_plan_opening(refined, units_per_metre, (float(centre[0]), float(centre[1])), angle))
    closed: list[dict[str, Any]] = []
    for item in result.closed:
        placed = _snap_to_face(np.asarray(item["center"], float), float(item["axis_angle"]), face_lines, snap)
        if placed is not None:  # a door candidate classified as no opening: the wall has to be closed over its span
            closed.append({"center": [round(float(placed[0][0]), 4), round(float(placed[0][1]), 4)], "axis_angle": round(float(placed[1]), 5), "width": round(float(item["width_m"]) * units_per_metre, 4), "thickness": round(float(item["wall_thickness_m"]) * units_per_metre, 4)})
    fills = host_side_spans(result.openings, load_refine_plan(cache_directory), config or load_openings_config(), [_ClosedSpan(item) for item in result.closed])
    info = {"sha256": sha, "applied": True, "mode": result.mode, "count": len(replaced), "dropped": len(dropped), "dropped_ids": dropped, "closed": closed, "fills": fills}
    _LOGGER.info("openings3d: applied %d refined openings (%d without a wall face dropped %s; was %d 2D openings)", len(replaced), len(dropped), dropped, len(openings))
    return replaced, info
