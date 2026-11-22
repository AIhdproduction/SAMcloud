"""Detection rules: windows (bands, mullions, glazing), doors (gap snapping), curtains, observation gating.

All inputs are the :class:`LineEvidence` grids of the wall lines; all thresholds come from
``config/bim_openings.yaml``.  Rules only emit an opening where the cells were *observed*: an unobserved
hole (no photograph saw it) stays unknown and is reported under ``unconfirmed``/``rejected``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np
import scipy.ndimage as ndi

from samcloud.bim.plan_model import BuildingPlan

from .config import Settings
from .evidence import LineEvidence
from .model import RefinedOpening
from .walls import WallLine, opening_placement


class SeedLike(Protocol):
    """A plan opening used as seed (PlanOpening)."""

    id: str
    kind: str
    center: list[float]
    width_m: float
    axis_angle: float
    wall_thickness_m: float
    host_wall_id: str | None


@dataclass
class Detection:
    openings: list[RefinedOpening]
    curtain_regions: list[dict[str, Any]] = field(default_factory=list)
    rejected: list[dict[str, Any]] = field(default_factory=list)
    closed: list[dict[str, Any]] = field(default_factory=list)  # door candidates classified as no opening (the wall closes)
    duplicates: list[RefinedOpening] = field(default_factory=list)  # candidates the dedupe rejected (a framed door among them is restored by classify)


def runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Inclusive (start, end) index pairs of the True runs of a 1-D mask."""
    index = np.flatnonzero(mask)
    if index.size == 0:
        return []
    breaks = np.flatnonzero(np.diff(index) > 1)
    starts = np.r_[index[0], index[breaks + 1]]
    ends = np.r_[index[breaks], index[-1]]
    return list(zip(starts.tolist(), ends.tolist()))


# --------------------------------------------------------------------------
# curtains
# --------------------------------------------------------------------------

def detect_curtains(ev: LineEvidence, config: Settings) -> list[dict[str, Any]]:
    """Contiguous furniture-prompt regions spanning (almost) floor to ceiling: curtains over windows."""
    cfg = config.curtains
    cell = config.grid.cell_m
    mask = (ndi.gaussian_filter(ev.furf, 1.0) > cfg.furniture_threshold) & ev.obs
    mask = ndi.binary_closing(mask, np.ones((5, 5)))
    mask = ndi.binary_opening(mask, np.ones((5, 5)))
    found = []
    for start, end in runs(mask.sum(0) >= cfg.min_column_m / cell):
        width = (end - start + 1) * cell
        if width < cfg.min_width_m:
            continue
        rows = np.flatnonzero(mask[:, start:end + 1].mean(1) > cfg.row_fraction)
        if rows.size == 0:
            continue
        v0, v1 = rows.min() * cell, (rows.max() + 1) * cell
        if v1 - v0 < cfg.min_height_m:
            continue
        found.append({
            "u0": round(start * cell, 2), "u1": round((end + 1) * cell, 2), "v0": round(float(v0), 2), "v1": round(float(v1), 2),
            "exterior": ev.line.kind == "exterior", "tall": bool(v0 < cfg.tall_sill_max_m and v1 > cfg.tall_head_min_m),
            "obs_frac": float(ev.obs[:, start:end + 1].mean()),
        })
    return found


# --------------------------------------------------------------------------
# windows
# --------------------------------------------------------------------------

def _window_candidates(ev: LineEvidence, config: Settings, rejected: list[dict[str, Any]]) -> list[dict[str, Any]]:
    cfg = config.windows
    cell = config.grid.cell_m
    morph = np.ones((int(cfg.morph_cells),) * 2)
    mask = (ev.pw > cfg.threshold) & ev.obs
    mask = ndi.binary_opening(ndi.binary_closing(mask, morph), morph)
    labels, count = ndi.label(mask)
    votes = ev.votes
    assert votes is not None
    candidates: list[dict[str, Any]] = []
    for index in range(1, count + 1):
        component = labels == index
        ys, xs = np.nonzero(component)
        if len(ys) * cell * cell < cfg.min_area_m2:
            continue
        columns = np.bincount(xs, minlength=ev.nu)
        rows = np.bincount(ys, minlength=ev.nv)
        cs = np.flatnonzero(columns >= cfg.extent_fraction * columns.max())
        rs = np.flatnonzero(rows >= cfg.extent_fraction * rows.max())
        x0, x1 = int(cs.min()), int(cs.max()) + 1
        y0, y1 = int(rs.min()), int(rs.max()) + 1
        if (x1 - x0) * cell < cfg.min_width_m or (y1 - y0) * cell < cfg.min_height_m:
            continue
        sub = (slice(y0, y1), slice(x0, x1))
        n_mean = float(votes.n[sub].mean())
        occ_mean = float(votes.occ[sub].mean())
        candidates.append({
            "x0": x0, "x1": x1, "y0": y0, "y1": y1, "mw": float(ev.pw[sub].mean()), "md": float(ev.pd[sub].mean()),
            "obsf": float(ev.obs[sub].mean()), "n": float(np.median(votes.n[sub])), "occ": occ_mean / max(1e-9, n_mean + occ_mean),
        })
    accepted = []
    for item in candidates:
        span = [round(item["x0"] * cell, 2), round(item["x1"] * cell, 2)]
        if item["md"] > item["mw"] + cfg.door_dominance:
            rejected.append({"line": ev.line.id, "what": "window-cand", "u": span, "reason": "door-class dominates window-class (door evidence)"})
            continue
        if item["obsf"] < cfg.min_observed_fraction or item["n"] < cfg.min_images:
            rejected.append({"line": ev.line.id, "what": "window-cand", "u": span, "reason": f"weak observation obs={item['obsf']:.2f} n={item['n']:.1f}"})
            continue
        if item["occ"] > cfg.max_occlusion:
            rejected.append({"line": ev.line.id, "what": "window-cand", "u": span, "reason": f"occluded {item['occ']:.2f}"})
            continue
        accepted.append(item)
    return accepted


def _merge_bands(candidates: list[dict[str, Any]], config: Settings) -> list[dict[str, Any]]:
    """Merge window rectangles with the same sill/head separated by a narrow gap (a mullion)."""
    cfg = config.windows
    cell = config.grid.cell_m
    bands: list[dict[str, Any]] = []
    for item in sorted(candidates, key=lambda c: c["x0"]):
        for band in bands:
            if abs(band["y0"] - item["y0"]) * cell < cfg.band_merge_dv_m and abs(band["y1"] - item["y1"]) * cell < cfg.band_merge_dv_m and (item["x0"] - band["x1"]) * cell < cfg.band_merge_gap_m:
                band["split_at"].append((band["x1"] + item["x0"]) / 2 * cell)
                band["x1"] = item["x1"]
                band["y0"] = min(band["y0"], item["y0"])
                band["y1"] = max(band["y1"], item["y1"])
                band["n"] = max(band["n"], item["n"])
                break
        else:
            new = dict(item)
            new["split_at"] = []
            bands.append(new)
    return bands


def _window_points_refine(ev: LineEvidence, band: dict[str, Any], config: Settings) -> tuple[float | None, float | None]:
    """Sill and head from the wall points that resume below/above the photographed window rectangle."""
    cfg = config.windows
    cell = config.grid.cell_m
    c0, c1 = band["x0"] + 2, band["x1"] - 2
    if c1 <= c0:
        return None, None
    row_solid = ev.solid[:, c0:c1].mean(1)
    fill = cfg.points_row_fill
    sill = head = None
    low = max(int((band["y0"] * cell - cfg.points_search_m) / cell), 0)
    high = min(int((band["y0"] * cell + cfg.points_search_m) / cell), ev.nv - 1)
    for i in range(high, low - 1, -1):
        if i >= 3 and row_solid[i] >= fill and row_solid[i - 1] >= fill and row_solid[i - 2] >= fill:
            sill = (i + 1) * cell
            break
    low = max(int((band["y1"] * cell - cfg.points_search_m) / cell), 0)
    high = min(int((band["y1"] * cell + cfg.points_search_m) / cell), ev.nv - 3)
    for i in range(low, high + 1):
        if row_solid[i] >= fill and row_solid[i + 1] >= fill and row_solid[i + 2] >= fill:
            head = i * cell
            break
    if sill is not None and head is not None:
        a, b = int(sill / cell), int(head / cell)
        if b > a and row_solid[a:b].mean() > cfg.points_zone_max_fill:
            return None, None
    return sill, head


def _window_edges_refine(ev: LineEvidence, band: dict[str, Any], config: Settings) -> tuple[float | None, float | None]:
    """Left/right edge (m) from the columns where wall points resume beside the photographed window rectangle."""
    cfg = config.windows
    cell = config.grid.cell_m
    r0, r1 = band["y0"] + 2, band["y1"] - 2
    if r1 - r0 < 3:
        return None, None
    column_solid = ev.solid[r0:r1].mean(0)
    fill = cfg.points_row_fill
    reach = int(cfg.points_search_m / cell)
    left = right = None
    for i in range(min(band["x0"] + reach, ev.nu - 1), max(band["x0"] - reach, 2) - 1, -1):
        if column_solid[i] >= fill and column_solid[i - 1] >= fill and column_solid[i - 2] >= fill:
            left = (i + 1) * cell
            break
    for i in range(max(band["x1"] - reach, 0), min(band["x1"] + reach, ev.nu - 3) + 1):
        if column_solid[i] >= fill and column_solid[i + 1] >= fill and column_solid[i + 2] >= fill:
            right = i * cell
            break
    if left is not None and right is not None and right - left < cfg.min_width_m:
        return None, None
    return left, right


def mullion_positions(ev: LineEvidence, band: dict[str, Any], v0: float, v1: float, config: Settings) -> list[tuple[float, list[str]]]:
    """Mullion bars inside a window rectangle: photo edges, solid point stripes, probability dips, band splits."""
    cfg = config.mullions
    cell = config.grid.cell_m
    x0, x1 = band["x0"], band["x1"]
    r0, r1 = int(v0 / cell) + 1, int(v1 / cell) - 1
    if r1 - r0 < cfg.min_rows or x1 - x0 < cfg.min_cols:
        return []
    import cv2

    gray = cv2.GaussianBlur(ev.gray.astype(np.float32), (3, 3), 0.7)
    gradient = np.abs(cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3))
    sub = gradient[r0:r1, x0:x1]
    observed = ev.obs[r0:r1, x0:x1]
    nx = x1 - x0
    edge = np.array([sub[:, i][observed[:, i]].mean() if observed[:, i].sum() > 0.5 * (r1 - r0) else np.nan for i in range(nx)])
    finite = np.isfinite(edge)
    threshold = np.nanmedian(edge) + cfg.edge_spread_std * np.nanstd(edge) if finite.sum() > 5 else np.inf
    median_gradient = float(np.nanmedian(sub[observed])) if observed.any() else np.inf
    frac = np.array([(sub[:, i][observed[:, i]] > median_gradient * cfg.edge_median_factor).mean() if observed[:, i].sum() > 0 else 0.0 for i in range(nx)])
    candidate = (np.nan_to_num(edge) > threshold) & (frac >= cfg.edge_min_frac)
    peaks = []
    for start, end in runs(candidate):
        segment = np.nan_to_num(edge[start:end + 1])
        peaks.append(start + int(segment.argmax()))
    photo_bars: list[list[int]] = []
    for peak in peaks:
        if photo_bars and (peak - photo_bars[-1][-1]) * cell <= cfg.photo_merge_m:
            photo_bars[-1].append(peak)
        else:
            photo_bars.append([peak])
    bars_photo = [float(np.mean(group)) for group in photo_bars]
    stripes = ev.solid[r0:r1, x0:x1].mean(0) > cfg.stripe_fill
    bars_points = [(s + e) / 2 for s, e in runs(stripes) if (e - s + 1) * cell <= cfg.max_bar_m]
    column_mean = ev.pw[r0:r1, x0:x1].mean(0)
    median = np.median(column_mean)
    bars_votes = [(s + e) / 2 for s, e in runs(column_mean < cfg.dip_fraction * median) if (e - s + 1) * cell <= cfg.max_bar_m]
    bars_split = [t / cell - x0 for t in band["split_at"]]
    every = [(p, "photo") for p in bars_photo] + [(p, "points") for p in bars_points] + [(p, "votes") for p in bars_votes] + [(p, "split") for p in bars_split]
    margin = cfg.edge_margin_m / cell
    every = sorted((p, s) for p, s in every if margin <= p <= nx - margin)
    clusters: list[dict[str, Any]] = []
    for position, source in every:
        if clusters and (position - clusters[-1]["ps"][-1]) * cell <= cfg.merge_m:
            clusters[-1]["ps"].append(position)
            clusters[-1]["src"].add(source)
        else:
            clusters.append({"ps": [position], "src": {source}})
    result = []
    for cluster in clusters:
        if "photo" in cluster["src"] or "split" in cluster["src"] or len(cluster["src"]) >= 2:
            result.append(((x0 + float(np.mean(cluster["ps"]))) * cell, sorted(cluster["src"])))
    return result


def detect_windows(ev: LineEvidence, config: Settings, rejected: list[dict[str, Any]]) -> None:
    """Window items (with refined sill/head and mullions) for one line; fills ``ev.curtains`` and ``ev.items``."""
    cell = config.grid.cell_m
    cfg = config.windows
    ev.curtains = detect_curtains(ev, config)
    ev.bands = _merge_bands(_window_candidates(ev, config, rejected), config)
    kept_bands = []
    for band in ev.bands:
        # A rectangle that reaches (or nearly reaches) a padded, artificial line end is cut there; the neighbouring line holds the whole window.
        margin = int(round(cfg.pad_edge_margin_m / cell))
        if (ev.line.pad_start_m > 0 and band["x0"] <= margin) or (ev.line.pad_end_m > 0 and band["x1"] >= ev.nu - margin):
            rejected.append({"line": ev.line.id, "what": "window-cand", "u": [round(band["x0"] * cell, 2), round(band["x1"] * cell, 2)], "reason": "cut by the padded line end (the neighbouring line has the whole window)"})
            continue
        kept_bands.append(band)
    ev.bands = kept_bands
    for band in ev.bands:
        photo0, photo1 = band["y0"] * cell, band["y1"] * cell
        points0, points1 = _window_points_refine(ev, band, config)

        def combine(photo: float, points: float | None) -> tuple[float, str]:
            if points is not None and abs(points - photo) <= cfg.photo_points_agree_m:
                return round(cfg.photo_weight * photo + cfg.points_weight * points, 3), "photo+points"
            return round(photo, 3), "photo"

        sill, source0 = combine(photo0, points0)
        head, source1 = combine(photo1, points1)
        source = "photo+points" if "photo+points" in (source0, source1) else "photo"
        if photo0 < cfg.floor_sill_m:
            sill = 0.0
        bars = mullion_positions(ev, band, sill if sill > 0 else 0.05, head, config)
        photo_u0, photo_u1 = band["x0"] * cell, band["x1"] * cell
        edge0, edge1 = _window_edges_refine(ev, band, config)
        u0 = round(cfg.photo_weight * photo_u0 + cfg.points_weight * edge0, 3) if edge0 is not None and abs(edge0 - photo_u0) <= cfg.photo_points_agree_m else photo_u0
        u1 = round(cfg.photo_weight * photo_u1 + cfg.points_weight * edge1, 3) if edge1 is not None and abs(edge1 - photo_u1) <= cfg.photo_points_agree_m else photo_u1
        margin = config.mullions.edge_margin_m  # the refined edges may have moved towards a bar found on the photograph extent
        bars = [(p, sources) for p, sources in bars if u0 + margin <= p <= u1 - margin]
        ev.items.append({
            "type": "window", "u0": u0, "u1": u1, "v0": sill, "v1": head, "src": source, "p0": points0, "p1": points1,
            "mull": bars, "mw": band["mw"], "md": band["md"], "n": band["n"], "obsf": band["obsf"], "occ": band["occ"],
        })
    extend_over_curtains(ev, config)


def extend_over_curtains(ev: LineEvidence, config: Settings) -> None:
    """A tall curtain between two windows of equal sill/head on an exterior wall hides one more window."""
    cfg = config.curtains
    windows = sorted((i for i in ev.items if i["type"] == "window"), key=lambda i: i["u0"])
    for curtain in ev.curtains:
        if not (curtain["exterior"] and curtain["tall"]):
            continue
        left = [i for i in windows if abs(i["u1"] - curtain["u0"]) <= cfg.join_distance_m]
        right = [i for i in windows if abs(i["u0"] - curtain["u1"]) <= cfg.join_distance_m]
        curtain["covers_windows"] = [round(i["u0"], 2) for i in windows if i["u0"] >= curtain["u0"] - 0.1 and i["u1"] <= curtain["u1"] + 0.1]
        if left and right and abs(left[0]["v0"] - right[0]["v0"]) < cfg.join_dv_m and abs(left[0]["v1"] - right[0]["v1"]) < cfg.join_dv_m:
            ev.items.append({
                "type": "window", "u0": left[0]["u1"], "u1": right[0]["u0"], "v0": (left[0]["v0"] + right[0]["v0"]) / 2, "v1": (left[0]["v1"] + right[0]["v1"]) / 2,
                "src": "photo", "p0": None, "p1": None, "mull": [], "mw": 0, "md": 0, "n": 0, "obsf": 0, "occ": 0, "curtain_extended": True,
            })


# --------------------------------------------------------------------------
# doors
# --------------------------------------------------------------------------

def lintel_head(solid: np.ndarray, c0: int, c1: int, cell: float, rows_from: float, fill: float) -> float | None:
    """Lowest height above ``rows_from`` where wall points resume over the gap columns."""
    sub = solid[:, c0:c1]
    if sub.shape[1] == 0:
        return None
    rows = sub.mean(1)
    for i in range(int(rows_from / cell), len(rows) - 2):
        if rows[i] >= fill and rows[i + 1] >= fill and rows[i + 2] >= fill:
            return i * cell
    return None


def gap_runs(ev: LineEvidence, config: Settings) -> list[tuple[int, int]]:
    """Columns where the wall points are missing between knee and shoulder height (floor gaps)."""
    cfg = config.doors
    cell = config.grid.cell_m
    rows = slice(int(cfg.gap_row_range_m[0] / cell), int(cfg.gap_row_range_m[1] / cell))
    hole = ev.solid[rows].mean(0) < cfg.gap_solid_max
    return runs(ndi.binary_closing(hole, np.ones(3)))


def evaluate_gap(ev: LineEvidence, c0: int, c1: int, config: Settings) -> dict[str, float]:
    """Door evidence inside a gap: point class fractions, photo door probability, see-through, observation."""
    cfg = config.doors
    cell = config.grid.cell_m
    r0, r1 = int(cfg.evidence_rows_m[0] / cell), int(cfg.evidence_rows_m[1] / cell)
    grids = ev.grids
    door_points = int(grids["door"][r0:r1, c0:c1].sum())
    window_points = int(grids["win"][r0:r1, c0:c1].sum())
    total = sum(int(grids[k][r0:r1, c0:c1].sum()) for k in ("door", "win", "wall", "other"))
    observed = ev.obs[r0:r1, c0:c1]
    photo = float(ev.pd[r0:r1, c0:c1][observed].mean()) if observed.sum() > cfg.min_evidence_cells else 0.0
    obs_fraction = float(observed.mean()) if observed.size else 0.0
    behind = 0.0
    if ev.votes is not None:
        rays = ev.votes.rays[r0:r1, c0:c1] > cfg.min_rays
        behind = float(ev.behind[r0:r1, c0:c1][rays].mean()) if rays.sum() > cfg.min_evidence_cells else 0.0
    return {"dpts": door_points, "dfrac": round(door_points / max(total, 1), 3), "wpts": window_points, "evb": round(photo, 3), "obsf": round(obs_fraction, 3), "behind": round(behind, 3)}


def lintel_fill(ev: LineEvidence, c0: int, c1: int, config: Settings) -> float:
    """Share of wall-point cells over the gap between the door-head search height and the top of the wall plane.

    A door stands under a lintel (wall points above it up to the ceiling): the fill is high.  An opening that reaches
    the ceiling (a passage, no lintel) has hardly any wall points there.
    """
    cfg = config.doors
    cell = config.grid.cell_m
    sub = ev.solid[int(cfg.head_search_from_m / cell):, c0 + 1:c1 - 1]
    return float(sub.mean()) if sub.size else float("nan")


def door_extent(ev: LineEvidence, c0: int, c1: int, config: Settings) -> tuple[float, float, float | None, float | None]:
    """(u0, u1, head from points, head from photographs) of a door gap."""
    cfg = config.doors
    cell = config.grid.cell_m
    head_points = lintel_head(ev.solid, c0 + 1, c1 - 1, cell, cfg.head_search_from_m, cfg.lintel_row_fill)
    head_photo = None
    sub = ev.pd[:, c0:c1]
    if sub.size:
        rows = np.flatnonzero(sub.mean(1) > cfg.photo_head_prob)
        if rows.size:
            head_photo = (rows.max() + 1) * cell
    if head_photo is not None and not (cfg.head_range_m[0] <= head_photo <= cfg.head_range_m[1]):
        head_photo = None
    return c0 * cell, c1 * cell, head_points, head_photo


def _line_of_point(evidence: dict[str, LineEvidence], point: list[float], plan: BuildingPlan, config: Settings, axis_angle: float | None = None) -> tuple[float, str, float] | None:
    """Wall line nearest to ``point`` (plan units) and the position along it.

    With ``axis_angle`` (the direction of the wall a seed door sits in) only lines running that way qualify: a door whose gap
    lies in a vertical wall is never measured on a horizontal wall that happens to pass nearby (its open leaf may lie along it).
    """
    upm = plan.units_per_metre
    tolerance = config.doors.line_distance_m
    sin_limit = math.sin(math.radians(config.doors.seed_axis_tolerance_deg))
    axis = None if axis_angle is None else np.array([math.cos(axis_angle), math.sin(axis_angle)])
    best = None
    for line_id, ev in evidence.items():
        line = ev.line
        if axis is not None and abs(float(line.d[0] * axis[1] - line.d[1] * axis[0])) > sin_limit:
            continue
        relative = np.asarray(point, float) - line.a
        u = float(relative @ line.d) / upm
        across = abs(float(relative @ line.n)) / upm
        if across < tolerance and -0.3 < u < line.length_m + 0.3 and (best is None or across < best[0]):
            best = (across, line_id, u)
    return best


def detect_doors(evidence: dict[str, LineEvidence], seeds: list[SeedLike], plan: BuildingPlan, config: Settings, rejected: list[dict[str, Any]]) -> list[Any]:
    """Doors from plan seeds snapped to point-cloud floor gaps plus unseeded gaps with door/see-through evidence.

    Returns the seeds that have no wall-plane data (short stubs) so the caller can keep them as plan gaps.
    """
    cfg = config.doors
    cell = config.grid.cell_m
    used: list[tuple[str, int, int]] = []
    orphans: list[Any] = []
    for seed in seeds:
        found = _line_of_point(evidence, seed.center, plan, config, seed.axis_angle)
        if found is None:
            orphans.append(seed)
            continue
        _, line_id, uc = found
        ev = evidence[line_id]
        plan_width = float(seed.width_m)
        centre_cell = int(uc / cell)
        gaps = [g for g in gap_runs(ev, config) if g[0] - cfg.seed_search_cells <= centre_cell <= g[1] + cfg.seed_search_cells]
        snapped = None
        if gaps:
            half = int(plan_width / cell)
            gap = max(gaps, key=lambda g: min(g[1], centre_cell + half) - max(g[0], centre_cell - half))
            gap_width = (gap[1] - gap[0] + 1) * cell
            if cfg.min_width_m <= gap_width <= cfg.max_width_m and abs(gap_width - plan_width) <= cfg.snap_max_width_diff_m and gap[0] - cfg.snap_centre_cells <= centre_cell <= gap[1] + cfg.snap_centre_cells:
                snapped = (gap[0], gap[1] + 1)
        if snapped is None:
            c0 = max(int(round((uc - plan_width / 2) / cell)), 0)
            c1 = min(int(round((uc + plan_width / 2) / cell)), ev.nu)
            width_source = "plan"
        else:
            c0, c1 = snapped
            width_source = "snapped"
        gap_ev = evaluate_gap(ev, c0, c1, config)
        u0, u1, head_points, head_photo = door_extent(ev, c0, c1, config)
        gap_ev["lintel"] = round(lintel_fill(ev, c0, c1, config), 3)
        ev.items.append({"type": "door", "u0": u0, "u1": u1, "head_pts": head_points, "head_ph": head_photo, "ev": gap_ev, "wsrc": width_source, "seed": seed.id, "plan_w": plan_width, "plan_kind": seed.kind})
        used.append((line_id, c0, c1))
    for line_id, ev in evidence.items():
        for gap in gap_runs(ev, config):
            c0, c1 = gap[0], gap[1] + 1
            width = (c1 - c0) * cell
            if any(l == line_id and min(c1, e1) - max(c0, e0) > 0 for l, e0, e1 in used):
                continue
            if any(i["type"] == "window" and min(i["u1"], c1 * cell) - max(i["u0"], c0 * cell) > cfg.floor_gap_window_overlap * width and i["v0"] < cfg.floor_gap_window_sill_m for i in ev.items):
                continue
            span = [round(c0 * cell, 2), round(c1 * cell, 2)]
            if width < cfg.min_width_m or width > cfg.max_width_m:
                if width >= cfg.min_width_m:
                    rejected.append({"line": line_id, "what": "floor gap", "u": span, "reason": f"width {width:.2f} m outside {cfg.min_width_m}-{cfg.max_width_m} (likely unobserved/occluded wall)"})
                continue
            if c0 <= cfg.end_margin_cells or c1 >= ev.nu - cfg.end_margin_cells:
                rejected.append({"line": line_id, "what": "floor gap", "u": span, "reason": "touches wall end (sliver / unobserved end)"})
                continue
            gap_ev = evaluate_gap(ev, c0, c1, config)
            door_evidence = gap_ev["dfrac"] >= cfg.door_fraction_min or (gap_ev["evb"] >= cfg.door_prob_min and gap_ev["behind"] >= cfg.behind_min)
            if gap_ev["obsf"] < cfg.min_observed_fraction:
                rejected.append({"line": line_id, "what": "floor gap", "u": span, "reason": f"weakly observed (B obs {gap_ev['obsf']}) -> unknown"})
                continue
            see_through = gap_ev["obsf"] >= cfg.see_through_obs and gap_ev["behind"] >= cfg.see_through_behind
            if not (door_evidence or see_through):
                rejected.append({"line": line_id, "what": "floor gap", "u": span, "reason": f"no door evidence and not seen through ({gap_ev})"})
                continue
            u0, u1, head_points, head_photo = door_extent(ev, c0, c1, config)
            gap_ev["lintel"] = round(lintel_fill(ev, c0, c1, config), 3)
            ev.items.append({"type": "door", "u0": u0, "u1": u1, "head_pts": head_points, "head_ph": head_photo, "ev": gap_ev, "wsrc": "gap", "seed": None, "plan_w": None, "plan_kind": "door" if door_evidence else "passage", "see_through": see_through})
    return orphans


def _room_polygons(plan: BuildingPlan | None) -> list[Any]:
    """Shapely polygons of the plan rooms (empty without a plan)."""
    if plan is None:
        return []
    from shapely.geometry import Polygon

    polygons = []
    for room in plan.rooms:
        ring = (room.polygon or {}).get("exterior") or []
        if len(ring) >= 3:
            polygon = Polygon(ring).buffer(0)
            if not polygon.is_empty:
                polygons.append(polygon)
    return polygons


def rooms_on_both_sides(line: WallLine, u0: float, u1: float, rooms: list[Any], probe_m: float, units_per_metre: float) -> bool:
    """True when a room lies on both sides of the wall at ``u0..u1`` (metres along the line): an interior wall, whatever the line kind says.

    A wall line merges collinear pieces, so a short interior wall that continues an exterior wall is part of a line that is
    called exterior.  The rooms of the plan show what stands on each side of the span.
    """
    if not rooms:
        return False
    from shapely.geometry import Point

    centre = line.a + line.d * ((u0 + u1) / 2.0) * units_per_metre
    reach = float(line.thickness) / 2.0 + probe_m * units_per_metre
    sides = []
    for sign in (1.0, -1.0):
        probe = Point(*(centre + sign * line.n * reach))
        sides.append(any(room.contains(probe) for room in rooms))
    return all(sides)


def classify_glazing_and_split(evidence: dict[str, LineEvidence], config: Settings, rejected: list[dict[str, Any]], plan: BuildingPlan | None = None) -> None:
    """Interior-wall window plausibility; doors with window evidence become glazed; windows split around doors."""
    cell = config.grid.cell_m
    win_cfg = config.windows
    cfg = config.doors
    rooms = _room_polygons(plan)
    upm = float(plan.units_per_metre) if plan is not None else 1.0
    for line_id, ev in evidence.items():
        kept = []
        for item in ev.items:
            if item["type"] == "window" and not item.get("curtain_extended"):
                if ev.line.kind != "exterior" and (item["v0"] < win_cfg.interior_min_sill_m or item["mw"] < win_cfg.interior_min_prob):
                    rejected.append({"line": line_id, "what": "window", "u": [round(item["u0"], 2), round(item["u1"], 2)], "reason": "window on interior wall with low sill / weak probability: likely door leaf/frame"})
                    continue
                if ev.line.kind == "exterior" and rooms_on_both_sides(ev.line, item["u0"], item["u1"], rooms, float(win_cfg.interior_probe_m), upm) and (item["v0"] < win_cfg.interior_min_sill_m or item["mw"] < win_cfg.interior_min_prob):
                    rejected.append({"line": line_id, "what": "window", "u": [round(item["u0"], 2), round(item["u1"], 2)], "reason": "window on a wall with a room on both sides (interior wall, line merged with an exterior wall) with low sill / weak probability: likely door leaf/frame"})
                    continue
            kept.append(item)
        ev.items = kept
    for line_id, ev in evidence.items():
        for door in [i for i in ev.items if i["type"] == "door"]:
            c0, c1 = int(round(door["u0"] / cell)), int(round(door["u1"] / cell))
            rows = slice(int(cfg.glazed_rows_m[0] / cell), int(cfg.glazed_rows_m[1] / cell))
            observed = ev.obs[rows, c0:c1]
            prob = float(ev.pw[rows, c0:c1][observed].mean()) if observed.sum() > cfg.min_evidence_cells else 0.0
            door["pw_in"] = prob
            door["glazed"] = prob >= cfg.glazed_window_prob
            new = []
            for window in [i for i in ev.items if i["type"] == "window" and not i.get("curtain_extended")]:
                overlap = min(window["u1"], door["u1"]) - max(window["u0"], door["u0"])
                if overlap > cfg.glazed_split_min_overlap * (door["u1"] - door["u0"]) and window["v0"] > 0.2:
                    if ev.line.kind == "exterior" and overlap >= cfg.window_absorbs_door_overlap * (door["u1"] - door["u0"]) and door in ev.items:
                        # A window rectangle with a sill above the floor spans the whole door gap on an exterior wall: the glazing starts
                        # above a parapet across the gap, so the gap is part of the window, never a door (a balcony door's glazing reaches the floor).
                        ev.items.remove(door)
                        rejected.append({"line": line_id, "what": "door", "u": [round(door["u0"], 2), round(door["u1"], 2)], "reason": f"door gap lies inside window rect {window['u0']:.2f}-{window['u1']:.2f} (sill {window['v0']:.2f} m) on an exterior wall: part of the window"})
                        break
                    ev.items.remove(window)
                    for a, b in ((window["u0"], door["u0"]), (door["u1"], window["u1"])):
                        if b - a >= cfg.glazed_split_min_piece_m:
                            piece = dict(window)
                            piece["u0"], piece["u1"] = a, b
                            piece["mull"] = [(p, s) for p, s in window["mull"] if a + cfg.window_split_margin_m <= p <= b - cfg.window_split_margin_m]
                            piece["split_from_door"] = door["seed"]
                            new.append(piece)
                    rejected.append({"line": line_id, "what": "window", "reason": f"window rect {window['u0']:.2f}-{window['u1']:.2f} overlapped door {door['u0']:.2f}-{door['u1']:.2f}; split around the door"})
            ev.items.extend(new)


# --------------------------------------------------------------------------
# records
# --------------------------------------------------------------------------

def _window_record(item: dict[str, Any], line: WallLine, plan: BuildingPlan, config: Settings) -> RefinedOpening:
    cfg = config.windows
    width = item["u1"] - item["u0"]
    centre, host, angle, thickness = opening_placement(line, (item["u0"] + item["u1"]) / 2, plan)
    sill, head, kind, glazed = item["v0"], item["v1"], "window", False
    if sill < cfg.floor_sill_m:
        sill = 0.0
        if width >= cfg.floor_glazing_min_width_m:
            glazed = True  # floor-to-ceiling glazing stays a window with sill 0
        elif width >= cfg.glazed_door_min_width_m:
            kind, glazed = "door", True  # glazed door
    mullions = [round(float(p), 3) for p, _ in item["mull"]]
    # mullion_u is measured from the left edge along the opening axis (the host piece direction), which may run against the wall line
    mirrored = float(np.dot(np.array([math.cos(angle), math.sin(angle)]), line.d)) < 0.0
    mullion_u = sorted(round(width - (m - item["u0"]), 3) if mirrored else round(m - item["u0"], 3) for m in mullions)
    if item.get("curtain_extended"):
        evidence, confidence = "curtain-extended", cfg.curtain_extended_confidence
    else:
        evidence = f"B window-prob {item['mw']:.2f}, obs {item['obsf']:.2f}, n {item['n']:.1f}, occ {item['occ']:.2f}; points {'yes' if item['src'] == 'photo+points' else 'sparse'}"
        confidence = round(min(0.99, item["mw"] * min(1, item["n"] / 4) * ((1 - cfg.confidence_points_bonus) + cfg.confidence_points_bonus * (item["src"] == "photo+points"))), 3)
    return RefinedOpening(
        id="", kind=kind, center=centre, width_m=round(width, 3), axis_angle=round(angle, 4), wall_thickness_m=round(thickness, 3), host_wall_id=host,
        sill_m=round(sill, 3), head_m=round(head, 3), height_source=item["src"], mullion_count=len(mullions),
        mullion_u=mullion_u, glazed=glazed, evidence=evidence, confidence=confidence, line=line.id, u0=round(item["u0"], 3), u1=round(item["u1"], 3),
    )


def _door_record(item: dict[str, Any], line: WallLine, plan: BuildingPlan, config: Settings) -> RefinedOpening:
    cfg = config.doors
    width = item["u1"] - item["u0"]
    centre, host, angle, thickness = opening_placement(line, (item["u0"] + item["u1"]) / 2, plan)
    gap = item["ev"]
    head, source = cfg.default_head_m, "default"
    head_points, head_photo = item["head_pts"], item["head_ph"]
    glazed = bool(item.get("glazed"))
    note = ""
    if head_points is not None and cfg.head_range_m[0] <= head_points <= cfg.head_range_m[1]:
        if head_photo is not None and abs(head_photo - head_points) <= cfg.head_agree_m:
            head, source = round(cfg.photo_weight * head_photo + cfg.points_weight * head_points, 3), "photo+points"
        else:
            head, source = round(head_points, 3), "points"
    elif head_photo is not None and (head_photo <= cfg.photo_head_max_m or glazed):
        head, source = round(head_photo, 3), "photo"
    if head_points is not None and head_points > cfg.head_range_m[1]:
        note = f"points lintel {head_points:.2f} > {cfg.photo_head_max_m} rejected"
    kind = item["plan_kind"] if item["plan_kind"] in ("door", "passage") else "door"
    if kind == "passage" and (gap["dfrac"] >= cfg.door_fraction_min or gap["evb"] >= cfg.door_prob_min):
        kind = "door"
    evidence = f"seed {item['seed'] or 'floor-gap'}; width {item['wsrc']}; door frac {gap['dfrac']}, B door-prob {gap['evb']}, behind {gap['behind']}, obs {gap['obsf']}, lintel {gap.get('lintel')}" + (f"; {note}" if note else "")
    confidence = round(min(0.99, cfg.confidence_base + (cfg.confidence_door_fraction if gap["dfrac"] >= cfg.door_fraction_min else 0) + (cfg.confidence_photo_door if gap["evb"] >= cfg.door_prob_min else 0) + (cfg.confidence_snapped if item["wsrc"] == "snapped" else 0) + (cfg.confidence_measured_head if source != "default" else 0)), 3)
    if glazed:
        evidence += f"; window-class {item.get('pw_in', 0):.2f} inside gap -> glazed door"
    return RefinedOpening(
        id="", kind=kind, center=centre, width_m=round(width, 3), axis_angle=round(angle, 4), wall_thickness_m=round(thickness, 3), host_wall_id=host,
        sill_m=0.0, head_m=head, height_source=source, mullion_count=0, mullion_u=[], glazed=glazed, evidence=evidence, confidence=confidence, line=line.id,
        u0=round(item["u0"], 3), u1=round(item["u1"], 3), plan_seed=item["seed"],
    )


def _distance_m(a: list[float], b: list[float], upm: float) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1]) / upm


def _deduplicate(records: list[RefinedOpening], plan: BuildingPlan, config: Settings, rejected: list[dict[str, Any]], duplicates: list[RefinedOpening] | None = None) -> list[RefinedOpening]:
    cfg = config.dedupe
    upm = plan.units_per_metre
    kept: list[RefinedOpening] = []
    for record in sorted(records, key=lambda r: -r.confidence):
        duplicate = False
        for other in kept:
            overlap = min(other.u1, record.u1) - max(other.u0, record.u0)
            if other.line == record.line and overlap > cfg.overlap_fraction * min(other.width_m, record.width_m):
                duplicate = True
            if _distance_m(other.center, record.center, upm) < cfg.centre_distance_m and other.kind == record.kind:
                duplicate = True
            if other.line != record.line and _same_family(other, record) and _parallel_overlap(other, record, upm) > cfg.overlap_fraction * min(other.width_m, record.width_m):
                duplicate = True  # one physical opening seen on two wall lines of the same wall (a wall split into collinear lines)
        if duplicate:
            rejected.append({"what": "duplicate", "center": record.center, "reason": "overlaps higher-confidence opening"})
            if duplicates is not None:
                duplicates.append(record)
            continue
        kept.append(record)
    return kept


def _same_family(first: RefinedOpening, second: RefinedOpening) -> bool:
    return (first.kind == "window") == (second.kind == "window")


def _parallel_overlap(first: RefinedOpening, second: RefinedOpening, units_per_metre: float) -> float:
    """Overlap (m) of two openings' spans along a common wall direction; 0 unless parallel and within one wall thickness."""
    a1, a2 = float(first.axis_angle), float(second.axis_angle)
    if abs(math.sin(a1 - a2)) > math.sin(math.radians(10.0)):
        return 0.0
    axis = np.array([math.cos(a1), math.sin(a1)])
    delta = (np.asarray(second.center, float) - np.asarray(first.center, float)) / units_per_metre
    across = abs(float(delta @ np.array([-axis[1], axis[0]])))
    if across > max(float(first.wall_thickness_m), float(second.wall_thickness_m), 0.1) / 2.0 + 0.15:
        return 0.0
    along = float(delta @ axis)
    return max(0.0, min(first.width_m / 2.0, along + second.width_m / 2.0) - max(-first.width_m / 2.0, along - second.width_m / 2.0))


def _number(records: list[RefinedOpening]) -> list[RefinedOpening]:
    ordered = sorted(records, key=lambda r: (r.center[1], r.center[0]))
    for index, record in enumerate(ordered, start=1):
        record.id = f"C{index:03d}"
    return ordered


def build_detection(evidence: dict[str, LineEvidence], orphan_seeds: list[Any], plan: BuildingPlan, config: Settings, rejected: list[dict[str, Any]]) -> Detection:
    """Turn the line items into the final opening records (dedupe, plan gaps without wall data, curtain regions)."""
    upm = plan.units_per_metre
    records: list[RefinedOpening] = []
    for ev in evidence.values():
        for item in ev.items:
            records.append(_window_record(item, ev.line, plan, config) if item["type"] == "window" else _door_record(item, ev.line, plan, config))
    duplicates: list[RefinedOpening] = []
    kept = _deduplicate(records, plan, config, rejected, duplicates)
    kept = _number(kept)
    for seed in orphan_seeds:
        near = [r for r in kept if _distance_m(r.center, list(seed.center), upm) < config.dedupe.plan_door_merge_distance_m]
        if near:
            rejected.append({"plan": seed.id, "what": "plan door", "reason": f"no wall-plane data; superseded by {near[0].id}"})
            continue
        kept.append(RefinedOpening(
            id="", kind=seed.kind, center=[round(float(seed.center[0]), 3), round(float(seed.center[1]), 3)], width_m=round(float(seed.width_m), 3),
            axis_angle=round(float(seed.axis_angle), 4), wall_thickness_m=round(float(seed.wall_thickness_m), 3), host_wall_id=seed.host_wall_id, sill_m=0.0,
            head_m=config.doors.default_head_m, height_source="default", evidence="plan gap only (host is a short stub, no wall-plane data)", confidence=config.doors.plan_gap_confidence, plan_seed=seed.id,
        ))
        rejected.append({"plan": seed.id, "what": "plan door", "reason": "kept as plan gap (low confidence)"})
        kept = _number(kept)
    curtains = []
    for ev in evidence.values():
        for curtain in ev.curtains:
            if not curtain["tall"] or not curtain["exterior"]:
                continue
            _, host, _, _ = opening_placement(ev.line, (curtain["u0"] + curtain["u1"]) / 2, plan)
            curtains.append({
                "wall": ev.line.id, "host_wall_id": host, "u_range": [curtain["u0"], curtain["u1"]], "v_range": [curtain["v0"], curtain["v1"]],
                "flag": "curtain_covered", "windows_inside": curtain.get("covers_windows", []), "obs_frac": round(curtain["obs_frac"], 2),
            })
    return Detection(kept, curtains, rejected, duplicates=duplicates)


def detect_openings(evidence: dict[str, LineEvidence], seeds: list[SeedLike], plan: BuildingPlan, config: Settings) -> Detection:
    """Run all photograph + point rules on the line evidence and return the refined openings."""
    rejected: list[dict[str, Any]] = []
    for ev in evidence.values():
        ev.items, ev.curtains, ev.bands = [], [], []
        detect_windows(ev, config, rejected)
    orphans = detect_doors(evidence, seeds, plan, config, rejected)
    classify_glazing_and_split(evidence, config, rejected, plan)
    return build_detection(evidence, orphans, plan, config, rejected)
