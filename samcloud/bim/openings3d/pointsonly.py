"""Points-only opening detection (no photographs): hole rectangles in the wall-point grid.

Used as the fallback when no CUDA device / no images are available and to list point-cloud windows that no
photograph confirmed.  Observation safety: a hole only becomes an opening when wall points *bound* it
(left/right/top for doors, three sides for windows) or when window/door class points fill it; a hole that
merely touches the wall end or has no bounding wall points is unobserved and stays unknown.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import scipy.ndimage as ndi

from samcloud.bim.plan_model import BuildingPlan

from .config import Settings
from .detect import Detection, SeedLike
from .evidence import LineEvidence
from .model import RefinedOpening
from .walls import opening_placement


def max_rectangle(mask: np.ndarray) -> tuple[int, tuple[int, int, int, int] | None]:
    """Largest axis-aligned all-True rectangle of ``mask`` indexed [u, v]: (area, (u0, u1, v0, v1))."""
    nu, nv = mask.shape
    best: tuple[int, tuple[int, int, int, int] | None] = (0, None)
    heights = np.zeros((nu, nv), int)
    for i in range(nu):
        heights[i] = np.where(mask[i], (heights[i - 1] if i else 0) + 1, 0)
    for i in range(nu):
        row = heights[i]
        for j in range(nv):
            if row[j] == 0:
                continue
            height = row[j]
            a = j
            while a > 0 and row[a - 1] >= height:
                a -= 1
            b = j
            while b < nv - 1 and row[b + 1] >= height:
                b += 1
            area = height * (b - a + 1)
            if area > best[0]:
                best = (area, (i - height + 1, i + 1, a, b + 1))
    return best


def _fraction(solid: np.ndarray, region: tuple[slice, slice]) -> float:
    return float(solid[region].mean()) if solid[region].size else 1.0


def _hole_rectangles(solid: np.ndarray, config: Settings) -> list[dict[str, Any]]:
    cfg = config.points_only
    cell = config.grid.cell_m
    nu, nv = solid.shape
    mask = ndi.binary_closing(~solid, np.ones((3, 3)), iterations=int(cfg.closing_iterations))
    mask = ndi.binary_opening(mask, np.ones((3, 3)))
    labels, count = ndi.label(mask)
    rectangles = []
    pad = 3
    for index in range(1, count + 1):
        component = labels == index
        for _ in range(int(cfg.max_rects_per_component)):
            area, rect = max_rectangle(component)
            if rect is None or area < cfg.min_rect_area_m2 / cell**2:
                break
            u0, u1, v0, v1 = rect
            component[u0:u1, v0:v1] = False
            if (u1 - u0) * cell < cfg.min_width_m or (v1 - v0) * cell < cfg.min_height_m:
                continue
            left = _fraction(solid, (slice(max(u0 - pad, 0), u0), slice(v0, v1))) if u0 > pad else 1.0
            right = _fraction(solid, (slice(u1, min(u1 + pad, nu)), slice(v0, v1))) if u1 < nu - pad else 1.0
            top = _fraction(solid, (slice(u0, u1), slice(v1, min(v1 + pad, nv)))) if v1 < nv - pad else 1.0
            bottom = _fraction(solid, (slice(u0, u1), slice(max(v0 - pad, 0), v0))) if v0 > pad else 1.0
            rectangles.append({"u0": u0, "u1": u1, "v0": v0, "v1": v1, "left": left, "right": right, "top": top, "bot": bottom, "edge": bool(u0 <= 2 or u1 >= nu - 2)})
    return rectangles


def _option(section: Settings, name: str, default: Any) -> Any:
    """Optional threshold (older config files / test overrides may not have it)."""
    try:
        return getattr(section, name)
    except AttributeError:
        return default


def _gap_runs(mask: np.ndarray, max_gap: int) -> list[tuple[int, int]]:
    """(start, stop) of every True run of ``mask``; gaps of up to ``max_gap`` cells do not end a run."""
    index = np.flatnonzero(mask)
    if index.size == 0:
        return []
    runs: list[tuple[int, int]] = []
    start = previous = int(index[0])
    for value in [*index[1:].tolist(), None]:
        if value is not None and value - previous <= max_gap + 1:
            previous = value
            continue
        runs.append((start, previous + 1))
        if value is not None:
            start = previous = value
    return runs


def _longest_run(mask: np.ndarray, max_gap: int) -> tuple[int, int] | None:
    """The longest run of :func:`_gap_runs`, None when ``mask`` is all False."""
    runs = _gap_runs(mask, max_gap)
    return max(runs, key=lambda run: run[1] - run[0]) if runs else None


def _window_extents(grids: dict[str, np.ndarray], rect: dict[str, Any], counts: dict[str, int], config: Settings) -> list[dict[str, float]]:
    """Glazing spans (u0, u1, v0, v1 in metres) of a hole dominated by window-class points; empty when it is not.

    The window-class points (frame and glass returns) give the windows; the hole rectangle can be larger (furniture and its scan
    shadow below the sill, a shelf beside the window, two windows whose holes merged).  Requires window points that clearly
    dominate door points and fill each glazing box densely, so a door, a glass door with its frame or a hole with stray window
    points never qualifies.
    """
    cfg = config.points_only
    cell = float(config.grid.cell_m)
    dominance = float(_option(cfg, "glazing_door_dominance", 0.0))
    if dominance <= 0.0 or counts["win"] <= dominance * max(counts["door"], 1):
        return []
    # The glazing may reach beyond the hole rectangle (a rectangle of an L-shaped hole, a sill hidden by furniture): it is searched
    # over the full height and half a metre beside the rectangle.
    pad = int(round(0.5 / cell))
    u0 = max(rect["u0"] - pad, 0)
    win = grids["win"][u0:min(rect["u1"] + pad, grids["win"].shape[0]), :] > 0
    if not win.any():
        return []
    min_cover = float(_option(cfg, "glazing_min_cover", 0.3))
    min_fill = float(_option(cfg, "glazing_min_fill", 0.5))
    column_rows = max(2, int(round(float(cfg.window_min_height_m) / cell * min_cover)))
    extents: list[dict[str, float]] = []
    for c0, c1 in _gap_runs(win.sum(1) >= column_rows, 2):
        rows = _longest_run(win[c0:c1].mean(0) >= min_cover, 2)
        if rows is None:
            continue
        columns = _longest_run(win[c0:c1, rows[0]:rows[1]].mean(1) >= min_cover, 2)
        if columns is None:
            continue
        a0, a1 = c0 + columns[0], c0 + columns[1]
        box = win[a0:a1, rows[0]:rows[1]]
        width, height = (a1 - a0) * cell, (rows[1] - rows[0]) * cell
        if float(box.mean()) < min_fill or width < float(cfg.min_width_m) or height < float(cfg.window_min_height_m) or width * height < float(cfg.min_rect_area_m2):
            continue
        if rows[1] * cell > float(cfg.window_max_head_m):
            continue
        if min(u0 + a1, rect["u1"]) - max(u0 + a0, rect["u0"]) <= 0:
            continue  # glazing beside the hole is another opening (found from its own hole)
        extents.append({"u0": (u0 + a0) * cell, "u1": (u0 + a1) * cell, "v0": rows[0] * cell, "v1": rows[1] * cell})
    return extents


def _furniture_shadow(grids: dict[str, np.ndarray], rect: dict[str, Any], counts: dict[str, int], config: Settings) -> bool:
    """True when a wall-bounded hole without any window-class point sits on furniture points (a scan shadow above a furniture top).

    Furniture standing in front of the wall hides the wall above its top edge from a scanner below that edge; the hidden stretch
    is bounded by wall on three sides like a window.  A window leaves window-class points (frame, glass); this hole leaves none,
    and furniture-class points fill the rows at its bottom edge.
    """
    cfg = config.points_only
    cell = float(config.grid.cell_m)
    limit = float(_option(cfg, "shadow_max_window_density", 0.0))
    if limit <= 0.0:
        return False
    area = max((rect["u1"] - rect["u0"]) * (rect["v1"] - rect["v0"]), 1)
    if counts["win"] / area > limit:
        return False
    depth = max(1, int(round(float(_option(cfg, "shadow_probe_m", 0.3)) / cell)))
    below = grids["other"][rect["u0"]:rect["u1"], max(rect["v0"] - depth, 0):rect["v0"] + depth]  # just below and in the hole's bottom rows
    if below.size == 0:
        return False
    # a column is covered by furniture when it holds a few points there (scattered stray points are no furniture top)
    return float((below.sum(1) >= int(_option(cfg, "shadow_column_points", 3))).mean()) >= float(_option(cfg, "shadow_min_cover", 0.3))


def detect_point_openings(ev: LineEvidence, config: Settings) -> list[dict[str, Any]]:
    """Windows/doors/passages of one line from the point grid alone (items in the layout of ``detect``)."""
    cfg = config.points_only
    cell = config.grid.cell_m
    grids = {name: array.T for name, array in ev.grids.items()}  # [u, v]
    solid = ev.solid.T
    candidates: list[dict[str, Any]] = []

    def evidence(rect: dict[str, Any]) -> dict[str, int]:
        sl = (slice(max(rect["u0"] - 2, 0), rect["u1"] + 2), slice(max(rect["v0"] - 2, 0), rect["v1"] + 2))
        return {name: int(grids[name][sl].sum()) for name in ("win", "door", "other", "wall")}

    door_min_width = float(_option(cfg, "door_min_width_m", 0.0))
    for rect in _hole_rectangles(solid, config):
        counts = evidence(rect)
        area = (rect["u1"] - rect["u0"]) * (rect["v1"] - rect["v0"])
        density_window, density_door = counts["win"] / area, counts["door"] / area
        v0, v1 = rect["v0"] * cell, rect["v1"] * cell
        width = (rect["u1"] - rect["u0"]) * cell
        bounded = int(rect["left"] > 0.5) + int(rect["right"] > 0.5) + int(rect["top"] > 0.5)
        kind = None
        if rect["edge"] and density_door <= cfg.window_evidence_density and density_window <= cfg.window_evidence_density:
            continue
        glazing = _window_extents(grids, rect, counts, config)
        if glazing:
            # The hole is dominated by window-class points: windows whose glazing (not the hole) gives the span.  Furniture in front of
            # the wall below the sill and its scan shadow merge with the glass into one hole that reaches the floor; that hole is no door.
            candidates.extend({"kind": "window", **extent, "ev": counts, "rect": rect} for extent in glazing)
            continue
        if v0 < cfg.door_floor_v_m:
            if width < door_min_width:
                continue  # narrower than any door: a wall projection, a niche or a gap between furniture, not a doorway
            v0 = 0.0
            if cfg.door_head_range_m[0] <= v1 <= cfg.door_head_range_m[1] and width <= cfg.door_max_width_m and ((bounded >= 2 and rect["top"] > 0.35) or density_door > cfg.window_evidence_density):
                kind = "door" if counts["door"] > cfg.door_min_points else "passage"
        elif cfg.window_sill_range_m[0] <= v0 <= cfg.window_sill_range_m[1] and (v1 - v0) >= cfg.window_min_height_m and v1 <= cfg.window_max_head_m:
            if bounded >= 3 and rect["bot"] > 0.35 and not _furniture_shadow(grids, rect, counts, config):
                kind = "window"
            elif density_window > cfg.window_evidence_density and bounded >= 2:
                kind = "window"
        if kind is None:
            continue
        candidates.append({"kind": kind, "u0": rect["u0"] * cell, "u1": rect["u1"] * cell, "v0": v0, "v1": v1, "ev": counts, "rect": rect})
    # window-class evidence blobs the hole rectangles missed (glass doors / mixed evidence)
    blob = ndi.uniform_filter(grids["win"].astype(float), 3) * 9 >= 4
    blob = ndi.binary_opening(ndi.binary_closing(blob, np.ones((3, 3)), iterations=2), np.ones((3, 3)))
    labels, count = ndi.label(blob)
    for index in range(1, count + 1):
        ii, jj = np.nonzero(labels == index)
        if len(ii) < 0.15 / cell**2:
            continue
        a0, a1 = np.percentile(ii, [2, 98])
        b0, b1 = np.percentile(jj, [2, 98])
        a0, a1, b0, b1 = int(a0), int(a1) + 1, int(b0), int(b1) + 1
        if (a1 - a0) * cell < 0.3 or (b1 - b0) * cell < cfg.min_height_m:
            continue
        overlaps = any(min(c["rect"]["u1"], a1) - max(c["rect"]["u0"], a0) > 0.3 * (a1 - a0) and min(c["rect"]["v1"], b1) - max(c["rect"]["v0"], b0) > 0.3 * (b1 - b0) for c in candidates)
        if overlaps or float((labels[a0:a1, b0:b1] == index).mean()) < 0.5:
            continue
        rect = {"u0": a0, "u1": a1, "v0": b0, "v1": b1}
        candidates.append({"kind": "window", "u0": a0 * cell, "u1": a1 * cell, "v0": b0 * cell, "v1": b1 * cell, "ev": evidence(rect), "rect": rect})
    # merge windows of one band across mullions, count internal solid stripes
    candidates.sort(key=lambda c: c["u0"])
    merged: list[dict[str, Any]] = []
    for item in candidates:
        if merged and item["kind"] == "window" and merged[-1]["kind"] == "window":
            previous = merged[-1]
            if abs(item["v0"] - previous["v0"]) <= cfg.merge_dv_m and abs(item["v1"] - previous["v1"]) <= cfg.merge_dv_m and item["u0"] - previous["u1"] < cfg.merge_gap_m:
                previous.setdefault("split_at", []).append((previous["u1"] + item["u0"]) / 2)
                previous["u1"] = item["u1"]
                previous["v0"] = min(previous["v0"], item["v0"])
                previous["v1"] = max(previous["v1"], item["v1"])
                continue
        merged.append(item)
    for item in merged:
        if item["kind"] != "window":
            item["mullions"] = []
            continue
        a0, a1 = int(item["u0"] / cell), int(item["u1"] / cell)
        b0, b1 = int(item["v0"] / cell), int(item["v1"] / cell)
        sub = solid[a0:a1, b0:b1]
        bars = list(item.get("split_at", []))
        if sub.size:
            column = sub.mean(1) > 0.6
            for s, e in [(s, e) for s, e in _runs(column)]:
                if (e - s + 1) * cell <= 0.15 and s > 0 and e < len(column) - 1:
                    bars.append((a0 + (s + e) / 2 + 0.5) * cell)
        item["mullions"] = sorted(bars)
    return merged


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    from .detect import runs

    return runs(mask)


def _spans_overlap(record: RefinedOpening, seed: SeedLike, units_per_metre: float) -> bool:
    """True when the plan seed lies on the record's wall (across < 0.5 m) and their spans along it overlap."""
    axis = np.array([np.cos(record.axis_angle), np.sin(record.axis_angle)])
    delta = (np.asarray(seed.center, float) - np.asarray(record.center, float)) / units_per_metre
    along, across = abs(float(delta @ axis)), abs(float(delta @ np.array([-axis[1], axis[0]])))
    return across < 0.5 and along < (float(record.width_m) + float(seed.width_m)) / 2.0


def _window_confidence(item: dict[str, Any], config: Settings) -> float:
    return round(config.points_only.confidence + 0.1 * min(1.0, (item["ev"]["win"] + item["ev"]["door"]) / 200.0), 3)


def points_only_detection(evidence: dict[str, LineEvidence], seeds: list[SeedLike], plan: BuildingPlan, config: Settings) -> Detection:
    """Fallback detection without photographs: heights from the points or defaults, low confidence."""
    from .detect import _deduplicate, _number

    rejected: list[dict[str, Any]] = []
    records: list[RefinedOpening] = []
    for ev in evidence.values():
        for item in detect_point_openings(ev, config):
            centre, host, angle, thickness = opening_placement(ev.line, (item["u0"] + item["u1"]) / 2, plan)
            width = item["u1"] - item["u0"]
            is_window = item["kind"] == "window"
            mirrored = float(np.dot(np.array([np.cos(angle), np.sin(angle)]), ev.line.d)) < 0.0
            mullions = sorted(round(width - (float(p) - item["u0"]), 3) if mirrored else round(float(p) - item["u0"], 3) for p in item["mullions"])
            records.append(RefinedOpening(
                id="", kind=item["kind"], center=centre, width_m=round(width, 3), axis_angle=round(angle, 4), wall_thickness_m=round(thickness, 3), host_wall_id=host,
                sill_m=round(item["v0"], 3) if is_window else 0.0, head_m=round(item["v1"], 3) if (is_window or item["v1"] <= config.doors.head_range_m[1]) else config.doors.default_head_m,
                height_source="points", mullion_count=len(mullions), mullion_u=mullions, glazed=False,
                evidence=f"points only (no photographs); hole bounded by wall points; evidence {item['ev']}", confidence=_window_confidence(item, config),
                line=ev.line.id, u0=round(item["u0"], 3), u1=round(item["u1"], 3),
            ))
    kept = _number(_deduplicate(records, plan, config, rejected))
    upm = plan.units_per_metre
    for seed in seeds:
        if seed.kind not in ("door", "passage"):
            continue
        near = [r for r in kept if np.hypot(r.center[0] - seed.center[0], r.center[1] - seed.center[1]) / upm < config.dedupe.plan_door_merge_distance_m]
        if near:
            # The measured hole IS the plan's door / passage: it keeps the seed, so the classification knows the 2D plan saw an
            # opening there (weak door evidence then keeps the safe default, a door).
            same = [r for r in near if r.kind in ("door", "passage") and r.plan_seed is None and _spans_overlap(r, seed, upm)]
            if same:
                min(same, key=lambda r: np.hypot(r.center[0] - seed.center[0], r.center[1] - seed.center[1])).plan_seed = seed.id
            continue
        kept.append(RefinedOpening(
            id="", kind=seed.kind, center=[round(float(seed.center[0]), 3), round(float(seed.center[1]), 3)], width_m=round(float(seed.width_m), 3), axis_angle=round(float(seed.axis_angle), 4),
            wall_thickness_m=round(float(seed.wall_thickness_m), 3), host_wall_id=seed.host_wall_id, sill_m=0.0, head_m=config.doors.default_head_m, height_source="default",
            evidence="plan gap only (no photographs, no point-cloud gap)", confidence=config.doors.plan_gap_confidence, plan_seed=seed.id,
        ))
    kept = _number(kept)
    return Detection(kept, [], rejected)
