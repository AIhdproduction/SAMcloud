"""Straight stairs from stair-class points: footprint, travel direction, risers, riser height, tread depth, from / to height.

A cluster of stair points is a stair only when its points rise steadily along its long axis (linear fit, slope 20..50 deg,
R^2 >= ``min_r2``), the heights show a regular riser period inside the plausible riser range, the tread levels follow each
other along the travel direction and the result obeys the stair rules (riser, tread, 2 r + g, width, at least
``min_risers`` risers).  Furniture with steps or scattered mislabelled points fail these tests.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from shapely.geometry import MultiPoint, Polygon
from shapely.ops import unary_union

from .common import STAIRS, LevelPoints, StructureSettings, contains


@dataclass(slots=True)
class Stair:
    start: tuple[float, float]  # foot of the first riser, centred across the width (metric plan)
    direction_deg: float  # travel direction, counter-clockwise from +x
    width_m: float
    riser_count: int
    riser_height_m: float
    tread_depth_m: float
    from_z_m: float
    to_z_m: float
    footprint: Polygon
    confidence: float
    evidence: dict[str, Any] = field(default_factory=dict)


def _riser_period(z: np.ndarray, low: float, high: float) -> tuple[float, float]:
    """Riser height from the autocorrelation of the height histogram (1 cm bins) and its normalised strength."""
    step = 0.005
    span = float(np.ptp(z))
    if span < 2 * low:
        return float("nan"), 0.0
    counts, _ = np.histogram(z, bins=int(math.ceil(span / step)) + 1, range=(float(z.min()), float(z.min()) + (int(math.ceil(span / step)) + 1) * step))
    signal = counts - counts.mean()
    energy = float(signal @ signal)
    if energy <= 0:
        return float("nan"), 0.0
    best_lag, best_value = 0, -np.inf
    for lag in range(int(round(low / step)), int(round(high / step)) + 1):
        if lag >= len(signal):
            break
        value = float(signal[:-lag] @ signal[lag:]) / energy
        if value > best_value:
            best_lag, best_value = lag, value
    return best_lag * step, max(best_value, 0.0)


def detect_stairs(
    points: LevelPoints, furniture: list[Polygon], settings: StructureSettings = StructureSettings(),
) -> tuple[list[Stair], list[dict[str, Any]]]:
    from sklearn.cluster import DBSCAN

    rejected: list[dict[str, Any]] = []
    stair_points = points.of((STAIRS,))
    if len(stair_points) < settings.stair_min_points:
        return [], rejected
    keys = np.floor(stair_points / 0.02).astype(np.int64)
    _, unique = np.unique(keys, axis=0, return_index=True)
    stair_points = stair_points[np.sort(unique)]
    labels = DBSCAN(eps=settings.stair_cluster_m, min_samples=10).fit_predict(stair_points[:, :2])
    furniture_union = unary_union(furniture) if furniture else None
    storey = float(points.storey_height_m)
    low_r, high_r = settings.stair_riser_m
    stairs: list[Stair] = []
    for label in sorted(set(labels.tolist()) - {-1}):
        cluster = stair_points[labels == label]
        info: dict[str, Any] = {"points": int(len(cluster)), "center": [round(float(v), 3) for v in cluster[:, :2].mean(axis=0)]}
        if len(cluster) < settings.stair_min_points:
            continue
        rectangle = MultiPoint([tuple(p) for p in cluster[:, :2]]).convex_hull.minimum_rotated_rectangle
        if rectangle.geom_type != "Polygon":
            rejected.append({**info, "reason": "degenerate"})
            continue
        corners = np.asarray(rectangle.exterior.coords)[:4]
        first, second = corners[1] - corners[0], corners[2] - corners[1]
        long_edge = first if np.hypot(*first) >= np.hypot(*second) else second
        width = float(min(np.hypot(*first), np.hypot(*second)))
        direction = long_edge / max(float(np.hypot(*long_edge)), 1e-12)
        centre = corners.mean(axis=0)
        s_all = (cluster[:, :2] - centre) @ direction
        z_all = cluster[:, 2]
        # Upper envelope along the long axis (2 cm stations, 90th height percentile): the tread surfaces.  A solid stair
        # also carries side and under faces at every height below its treads; they must not enter the step analysis.
        station = np.floor((s_all - s_all.min()) / 0.02).astype(int)
        tops = np.full(int(station.max()) + 1, np.nan)
        order = np.argsort(station, kind="stable")
        bounds = np.flatnonzero(np.diff(station[order])) + 1
        for group in np.split(order, bounds):
            if len(group) >= 3:
                tops[station[group[0]]] = float(np.percentile(z_all[group], 90))
        valid = np.isfinite(tops)
        if valid.sum() < 10:
            rejected.append({**info, "reason": "no_steady_rise"})
            continue
        envelope_s = s_all.min() + (np.flatnonzero(valid) + 0.5) * 0.02
        envelope_z = tops[valid]
        slope, intercept = np.polyfit(envelope_s, envelope_z, 1)
        top_mask = np.zeros(len(z_all), dtype=bool)
        top_mask[np.isfinite(tops[station])] = z_all[np.isfinite(tops[station])] >= tops[station][np.isfinite(tops[station])] - 0.03
        if slope < 0:
            direction, s_all, envelope_s, slope = -direction, -s_all, -envelope_s, -slope
            intercept = float(np.polyfit(envelope_s, envelope_z, 1)[1])
        predicted = slope * envelope_s + intercept
        r2 = 1.0 - float(((envelope_z - predicted) ** 2).sum()) / max(float(((envelope_z - envelope_z.mean()) ** 2).sum()), 1e-12)
        s, z = s_all[top_mask], z_all[top_mask]
        angle = math.degrees(math.atan(slope))
        info.update(width_m=round(width, 3), slope_deg=round(angle, 1), r2=round(r2, 3))
        if not (20.0 <= angle <= 50.0) or r2 < settings.stair_min_r2:
            rejected.append({**info, "reason": "no_steady_rise"})
            continue
        if not (settings.stair_min_width_m <= width <= settings.stair_max_width_m):
            rejected.append({**info, "reason": "width"})
            continue
        riser, strength = _riser_period(z, low_r, high_r)
        info.update(riser_m=round(riser, 3) if math.isfinite(riser) else None, period_strength=round(strength, 3))
        if not math.isfinite(riser) or strength < 0.2:
            rejected.append({**info, "reason": "no_riser_period"})
            continue
        phase = np.angle(np.exp(2j * np.pi * z / riser).mean()) / (2 * np.pi) * riser  # circular mean of the tread heights
        level_index = np.round((z - phase) / riser).astype(int)
        levels = sorted(set(level_index.tolist()))
        level_z = {k: phase + k * riser for k in levels}
        counts = {k: int((level_index == k).sum()) for k in levels}
        supported = [k for k in levels if counts[k] >= max(5, 0.02 * len(z))]
        if len(supported) < 2:
            rejected.append({**info, "reason": "too_few_treads"})
            continue
        lowest, highest = min(supported), max(supported)
        # a level at the floor (or at the upper floor) is the floor next to the stair, not a tread
        if abs(level_z[lowest]) < 0.5 * riser:
            lowest += 1
        if abs(level_z[highest] - storey) < 0.5 * riser:
            highest -= 1
        treads = list(range(lowest, highest + 1))
        occupied = [k for k in treads if k in supported]
        share = len(occupied) / max(len(treads), 1)
        info["level_share"] = round(share, 3)
        if len(treads) < settings.stair_min_risers - 1 or share < settings.stair_min_level_share:
            rejected.append({**info, "reason": "too_few_treads"})
            continue
        means = np.asarray([float(np.median(s[level_index == k])) for k in occupied])
        if len(means) >= 3:
            ranks = np.argsort(np.argsort(means))
            monotonic = float(np.corrcoef(ranks, np.arange(len(means)))[0, 1])
        else:
            monotonic = 1.0 if np.all(np.diff(means) > 0) else -1.0
        info["monotonic"] = round(monotonic, 3)
        if monotonic < 0.9:
            rejected.append({**info, "reason": "treads_not_in_sequence"})
            continue
        tread_mask = np.isin(level_index, treads)
        s_lo, s_hi = (float(v) for v in np.percentile(s[tread_mask], [1, 99]))
        n_treads = len(treads)
        tread = (s_hi - s_lo) / n_treads
        risers = n_treads + 1
        from_z = level_z[lowest] - riser
        to_z = from_z + risers * riser
        info.update(tread_m=round(tread, 3), risers=risers, from_z_m=round(from_z, 3), to_z_m=round(to_z, 3))
        low_g, high_g = settings.stair_tread_m
        low_c, high_c = settings.stair_comfort_m
        if not (low_r <= riser <= high_r) or not (low_g <= tread <= high_g) or not (low_c <= 2 * riser + tread <= high_c) or risers < settings.stair_min_risers:
            rejected.append({**info, "reason": "implausible_dimensions"})
            continue
        normal = np.asarray((-direction[1], direction[0]))
        t = (cluster[:, :2] - centre) @ normal
        width = float(np.percentile(t, 99.5) - np.percentile(t, 0.5))  # the hull rectangle is widened by the side-face noise
        t_mid = 0.5 * (float(np.percentile(t, 1)) + float(np.percentile(t, 99)))
        start = centre + direction * s_lo + normal * t_mid
        run = n_treads * tread
        half = width / 2.0
        footprint = Polygon([tuple(start + direction * a + normal * b) for a, b in ((0.0, -half), (run, -half), (run, half), (0.0, half))])
        if furniture_union is not None and not furniture_union.is_empty:
            overlap = footprint.intersection(furniture_union).area / max(footprint.area, 1e-9)
            info["furniture_overlap"] = round(float(overlap), 3)
            if overlap > 0.5:
                rejected.append({**info, "reason": "in_furniture"})
                continue
        confidence = float(np.clip(r2, 0, 1) * np.clip(strength * 2.0, 0, 1) * share)
        stairs.append(Stair(
            start=(float(start[0]), float(start[1])), direction_deg=float(math.degrees(math.atan2(direction[1], direction[0]))), width_m=width,
            riser_count=int(risers), riser_height_m=float(riser), tread_depth_m=float(tread), from_z_m=float(from_z), to_z_m=float(to_z),
            footprint=footprint, confidence=round(confidence, 3), evidence=info,
        ))
    return stairs, rejected


def points_in_stairs(stairs: list[Stair], xy: np.ndarray) -> np.ndarray:
    if not stairs or len(xy) == 0:
        return np.zeros(len(xy), dtype=bool)
    return contains(unary_union([stair.footprint for stair in stairs]), xy)
