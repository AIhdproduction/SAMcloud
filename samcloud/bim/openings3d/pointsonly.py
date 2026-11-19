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
        if v0 < cfg.door_floor_v_m:
            v0 = 0.0
            if cfg.door_head_range_m[0] <= v1 <= cfg.door_head_range_m[1] and width <= cfg.door_max_width_m and ((bounded >= 2 and rect["top"] > 0.35) or density_door > cfg.window_evidence_density):
                kind = "door" if counts["door"] > cfg.door_min_points else "passage"
        elif cfg.window_sill_range_m[0] <= v0 <= cfg.window_sill_range_m[1] and (v1 - v0) >= cfg.window_min_height_m and v1 <= cfg.window_max_head_m:
            if bounded >= 3 and rect["bot"] > 0.35:
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
            continue
        kept.append(RefinedOpening(
            id="", kind=seed.kind, center=[round(float(seed.center[0]), 3), round(float(seed.center[1]), 3)], width_m=round(float(seed.width_m), 3), axis_angle=round(float(seed.axis_angle), 4),
            wall_thickness_m=round(float(seed.wall_thickness_m), 3), host_wall_id=seed.host_wall_id, sill_m=0.0, head_m=config.doors.default_head_m, height_source="default",
            evidence="plan gap only (no photographs, no point-cloud gap)", confidence=config.doors.plan_gap_confidence, plan_seed=seed.id,
        ))
    kept = _number(kept)
    return Detection(kept, [], rejected)
