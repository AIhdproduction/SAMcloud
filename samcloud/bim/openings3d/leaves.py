"""Open door leaf: hinge, opening angle and swing side from door-class points (hinge-ray search).

An open leaf is a thin, door-class, door-high plane that stands on the hinge line of the door (at one of the two
jambs).  For both jambs the search tries hinge points within ``leaf.hinge_search_m`` of the jamb and rays every
``leaf.angle_step_deg`` degrees; the points inside a ``leaf.band_m`` wide band along a ray are binned along it and
the longest contiguous run of populated bins is the leaf.  A leaf must be 0.5-1.3 m long, reach from the lower
part of the door to head height and carry enough points, and it must have opened by at least ``leaf.min_angle_deg``
(a smaller angle is the closed leaf in the wall plane, not an open one).

Coordinates are metres in the plan frame; ``hinge`` / ``free`` are plan points.  ``hinge_side`` is ``"left"``
when the hinge is at the ``-axis`` end of the opening (``axis = (cos a, sin a)``), ``swing_side`` is +1 when the
free end is on the ``+normal`` side (``normal = (-sin a, cos a)``), the conventions of the plan model.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .config import Settings


@dataclass(frozen=True)
class Leaf:
    hinge_side: str  # left | right (see module doc)
    swing_side: int  # +1 / -1 (side of +normal the leaf swings to)
    angle_deg: float  # 0 = closed in the opening, 90 = square to the wall, 180 = flat against the wall
    length_m: float
    points: int
    z_min: float
    z_max: float
    hinge: tuple[float, float]  # plan metres
    free: tuple[float, float]
    hinge_offset_m: float  # distance of the jamb from the leaf (perpendicular to the leaf plane, in front of the hinge: the jamb itself)

    def to_dict(self) -> dict[str, object]:
        return {
            "hinge_side": self.hinge_side, "swing_side": self.swing_side, "angle_deg": round(self.angle_deg, 1), "length_m": round(self.length_m, 3), "points": self.points,
            "z_min": round(self.z_min, 2), "z_max": round(self.z_max, 2), "hinge": [round(self.hinge[0], 3), round(self.hinge[1], 3)], "free": [round(self.free[0], 3), round(self.free[1], 3)],
            "hinge_offset_m": round(self.hinge_offset_m, 3),
        }


def _run_length(counts: np.ndarray, min_points: int) -> np.ndarray:
    """Per row of ``counts``: bins reached by the first run of populated bins (one empty bin may be bridged)."""
    ok = counts >= min_points
    result = np.zeros(len(ok), dtype=int)
    for index, row in enumerate(ok):
        reach, gap = 0, 0
        for position, populated in enumerate(row):
            if populated:
                reach, gap = position + 1, 0
            else:
                gap += 1
                if gap > 1:
                    break
        result[index] = reach
    return result


def _jamb_distance(jamb: np.ndarray, hinge: np.ndarray, direction: np.ndarray) -> float:
    """Distance from the jamb to the leaf ray (hinge point + direction): perpendicular, or to the hinge when the jamb lies behind it.

    The fitted hinge point can slide along the leaf plane (every point of the extended leaf line fits equally), so the
    perpendicular distance, not the distance of the fitted hinge, says which jamb the leaf stands at.
    """
    offset = jamb - hinge
    along = float(offset @ direction)
    perpendicular = abs(float(-offset[0] * direction[1] + offset[1] * direction[0]))
    return perpendicular if along >= 0.0 else float(np.hypot(*offset))


def find_leaf(points: np.ndarray, centre: tuple[float, float], axis_angle: float, width_m: float, config: Settings) -> Leaf | None:
    """Best open leaf hinged at one jamb of the opening (``points`` = (N, 3) plan x, y metres and height)."""
    cfg = config.leaf
    axis = np.asarray((math.cos(axis_angle), math.sin(axis_angle)))
    normal = np.asarray((-axis[1], axis[0]))
    centre_xy = np.asarray(centre, float)
    relative = points[:, :2] - centre_xy
    distance = np.hypot(relative[:, 0], relative[:, 1])
    selected = distance < float(cfg.search_radius_m)
    if selected.sum() > int(cfg.max_points):
        indices = np.flatnonzero(selected)
        selected[:] = False
        selected[indices[np.linspace(0, len(indices) - 1, int(cfg.max_points)).astype(int)]] = True
    if selected.sum() < int(cfg.min_points):
        return None
    local = np.column_stack((relative[selected] @ axis, relative[selected] @ normal))
    heights = points[selected, 2]
    step = float(cfg.angle_step_deg)
    angles = np.radians(np.arange(0.0, 360.0, step))
    directions = np.column_stack((np.cos(angles), np.sin(angles)))  # (A, 2)
    bin_m = float(cfg.bin_m)
    bins = int(math.ceil(float(cfg.max_length_m) / bin_m)) + 1
    offsets = np.arange(-float(cfg.hinge_search_m), float(cfg.hinge_search_m) + 1e-9, float(cfg.hinge_step_m))
    candidates: list[Leaf] = []
    for side_name, sign in (("left", -1), ("right", 1)):
        best: tuple[tuple[float, int], dict[str, float | int]] | None = None
        jamb = np.asarray((sign * width_m / 2.0, 0.0))
        for hu in offsets:
            for hn in offsets:
                hinge = jamb + np.asarray((hu, hn))
                rel = local - hinge
                radius = np.hypot(rel[:, 0], rel[:, 1])
                near = radius < float(cfg.max_length_m) + 0.05
                if near.sum() < int(cfg.min_points):
                    continue
                rr = rel[near]
                along = rr @ directions.T  # (M, A)
                perp = np.abs(-rr[:, 0:1] * directions[None, :, 1] + rr[:, 1:2] * directions[None, :, 0])
                inside = (perp < float(cfg.band_m)) & (along > 0.03) & (along < float(cfg.max_length_m))
                totals = inside.sum(0)
                if totals.max() < int(cfg.min_points):
                    continue
                index = np.clip((along / bin_m).astype(int), 0, bins - 1)
                flat = (np.arange(len(directions))[None, :] * bins + index)[inside]
                counts = np.bincount(flat, minlength=len(directions) * bins).reshape(len(directions), bins)
                reach = _run_length(counts, int(cfg.bin_min_points))
                for a in np.flatnonzero(totals >= int(cfg.min_points)):
                    key = (float(reach[a] * bin_m), int(totals[a]))
                    if best is None or key > best[0]:
                        best = (key, {"a": int(a), "hu": float(hu), "hn": float(hn), "len": key[0], "n": key[1]})
        if best is None:
            continue
        info = best[1]
        direction = directions[int(info["a"])]
        hinge = jamb + np.asarray((float(info["hu"]), float(info["hn"])))
        rel = local - hinge
        along = rel @ direction
        perp = np.abs(-rel[:, 0] * direction[1] + rel[:, 1] * direction[0])
        inside = (perp < float(cfg.band_m)) & (along > 0.03) & (along < float(cfg.max_length_m))
        z_low, z_high = (float(np.percentile(heights[inside], q)) for q in (2, 98))
        length = float(info["len"])
        into = np.asarray((-sign, 0.0))
        angle = math.degrees(math.acos(float(np.clip(direction @ into, -1.0, 1.0))))
        free = hinge + direction * length
        leaf_like = (
            float(cfg.min_length_m) <= length <= float(cfg.max_length_m)
            and z_low < float(cfg.max_foot_m) and z_high > float(cfg.min_top_m)
            and int(info["n"]) >= int(cfg.min_leaf_points)
            and angle >= float(cfg.min_angle_deg)
            and abs(float(free[1] - hinge[1])) >= float(cfg.min_swing_m)
        )
        if not leaf_like:
            continue
        to_plan = lambda p: tuple(float(v) for v in (centre_xy + axis * p[0] + normal * p[1]))  # noqa: E731
        candidates.append(Leaf(
            hinge_side=side_name, swing_side=1 if float(free[1] - hinge[1]) > 0 else -1, angle_deg=angle, length_m=length, points=int(info["n"]), z_min=z_low, z_max=z_high,
            hinge=to_plan(hinge), free=to_plan(free), hinge_offset_m=_jamb_distance(jamb, hinge, direction),
        ))
    return max(candidates, key=lambda leaf: leaf.points) if candidates else None
