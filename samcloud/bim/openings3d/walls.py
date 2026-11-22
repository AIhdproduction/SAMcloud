"""Wall lines of a BuildingPlan (collinear pieces bridged over their openings) and the per-face grids on them.

A *wall line* merges collinear plan wall pieces whose gaps are openings, so a window or door gap is a
hole inside one line.  Each line has two faces (side +1 / -1); photographs are voted per face and the
two faces are summed again before detection.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from samcloud.bim.plan_model import BuildingPlan, PlanWall
from samcloud.dxf.las_source import Frame

from .config import Settings


SYNTHETIC_WALL_EVIDENCE = frozenset({"door-host"})


@dataclass(frozen=True)
class WallLine:
    """Merged collinear wall. ``a`` is on the centreline at u = 0; lengths in plan units unless ``*_m``."""

    id: str
    pieces: tuple[str, ...]
    a: np.ndarray
    d: np.ndarray
    n: np.ndarray  # left normal (-dy, dx); face side s points along s * n
    length: float
    length_m: float
    thickness: float  # plan units
    kind: str
    pad_start_m: float = 0.0  # grid length (m) that continues into a gap before the first wall piece
    pad_end_m: float = 0.0  # same after the last wall piece


@dataclass(frozen=True)
class WallFace:
    """One side of a wall line: ``n`` points from the wall into the room that is photographed."""

    line: WallLine
    side: int

    @property
    def n(self) -> np.ndarray:
        return self.line.n * self.side

    @property
    def key(self) -> tuple[str, int]:
        return (self.line.id, self.side)


def plan_frame(plan: BuildingPlan) -> Frame:
    data = plan.frame
    return Frame(origin=np.asarray(data["origin"], float), scale_factor=float(data["scale_factor"]), rotation=np.asarray(data["rotation"], float))


def storey_height_m(plan: BuildingPlan) -> float:
    if plan.ceiling_z is None:
        return 2.6
    return float(plan.ceiling_z - plan.floor_z) / plan.units_per_metre


def wall_lines(plan: BuildingPlan, config: Settings) -> list[WallLine]:
    """Merge collinear wall pieces into wall lines (gaps up to ``walls.max_gap_m`` are bridged)."""
    upm = plan.units_per_metre
    cfg = config.walls
    pieces: list[dict[str, Any]] = []
    for wall in plan.walls:
        if wall.evidence in SYNTHETIC_WALL_EVIDENCE:
            continue  # door-host pieces are only hosts the wall stage builds for doors without a wall; they are no measured wall to scan
        a, b = np.asarray(wall.centreline[0], float), np.asarray(wall.centreline[1], float)
        length = float(np.linalg.norm(b - a))
        if length < cfg.min_piece_length_m * upm:
            continue
        d = (b - a) / length
        if d[0] < -1e-9 or (abs(d[0]) < 1e-9 and d[1] < 0):
            d, a, b = -d, b, a
        pieces.append({"id": wall.id, "a": a, "b": b, "d": d, "t": float(wall.thickness_m), "kind": wall.kind})
    lines: list[dict[str, Any]] = []
    sin_limit = math.sin(math.radians(cfg.merge_angle_deg))
    for piece in pieces:
        for line in lines:
            normal = np.array([-line["d"][1], line["d"][0]])
            if abs(float(line["d"][0] * piece["d"][1] - line["d"][1] * piece["d"][0])) < sin_limit and abs(float((piece["a"] - line["a"]) @ normal)) < cfg.merge_offset_m * upm:
                line["pieces"].append(piece)
                break
        else:
            lines.append({"a": piece["a"], "d": piece["d"], "pieces": [piece]})
    result: list[WallLine] = []
    for line in lines:
        d = line["d"]
        n0 = np.array([-d[1], d[0]])
        spans = sorted([((p["a"] - line["a"]) @ d, (p["b"] - line["a"]) @ d, p) for p in line["pieces"]], key=lambda item: item[0])
        groups = [[spans[0]]]
        for span in spans[1:]:
            if span[0] - max(item[1] for item in groups[-1]) > cfg.max_gap_m * upm:
                groups.append([span])
            else:
                groups[-1].append(span)
        for index, group in enumerate(groups):
            s0 = min(item[0] for item in group)
            s1 = max(item[1] for item in group)
            length = s1 - s0
            if length < cfg.min_line_length_m * upm:
                continue
            offset = float(np.mean([(item[2]["a"] - line["a"]) @ n0 for item in group]))
            # The wall pieces next to a gap end at an opening (a window leaf that is not drawn as wall reaches beyond
            # the piece).  A gap longer than ``max_gap_m`` is not bridged, so the two groups become separate lines;
            # each then continues up to ``end_pad_m`` into the gap (the whole gap when it is shorter), the two lines
            # overlap and an opening in the gap lies completely on both.  Whether a gap is just below or just above the
            # bridge limit then does not decide where a window is cut.  Free ends and wall joints have no neighbour on
            # the same line and are not padded.
            limit = cfg.end_pad_m * upm
            pad_start = pad_end = 0.0
            if index > 0:
                pad_start = min(limit, s0 - max(item[1] for item in groups[index - 1]))
                s0 -= pad_start
                length += pad_start
            if index < len(groups) - 1:
                pad_end = min(limit, min(item[0] for item in groups[index + 1]) - s1)
                s1 += pad_end
                length += pad_end
            origin = line["a"] + d * s0 + n0 * offset
            thickness = float(np.median([item[2]["t"] for item in group])) * upm
            ids = [item[2]["id"] for item in group]
            kind = "exterior" if any(item[2]["kind"] == "exterior" for item in group) else group[0][2]["kind"]
            line_id = ids[0] + (f"+{len(ids) - 1}" if len(ids) > 1 else "")
            result.append(WallLine(line_id, tuple(ids), origin, d, n0, float(length), float(length) / upm, thickness, kind, pad_start / upm, pad_end / upm))
    return _extend_to_seed_gaps(result, plan, config)


def _extend_to_seed_gaps(lines: list[WallLine], plan: BuildingPlan, config: Settings) -> list[WallLine]:
    """Continue a wall line over a plan door/passage gap that lies beyond its end (a gap at the end of a wall).

    A door in the last stretch of a wall (the plan wall ends at the jamb, the gap runs on to the crossing wall) has no wall line
    around it, so the door would be measured on whatever other wall passes nearby.  The collinear line is continued to the far edge
    of the gap (at most ``walls.seed_extend_max_m``) and the continuation counts as padding (an artificial end, like a bridged gap).
    """
    cfg = config.walls
    upm = plan.units_per_metre
    sin_limit = math.sin(math.radians(cfg.merge_angle_deg))
    door_reach = float(cfg.seed_extend_max_m) * upm
    # A plan window in the last stretch of a wall (the wall piece ends at its jamb, the window runs on to the corner) is continued
    # the same way; without a wall line over it the window has no wall-plane grid and is lost in the points-only refinement.
    window_reach = float(getattr(cfg, "seed_extend_window_max_m", 0.0) or 0.0) * upm
    result = list(lines)
    seeds = sorted((o for o in plan.openings if o.kind in ("door", "passage") or (o.kind == "window" and window_reach > 0.0)), key=lambda o: o.id)
    # Windows side by side in the last stretch of a wall extend the line one after the other: repeat until nothing changes.
    for _ in range(max(1, len(seeds))):
        if not _extend_once(result, seeds, upm, sin_limit, door_reach, window_reach, cfg):
            break
    return result


def _extend_once(result: list[WallLine], seeds: list[Any], upm: float, sin_limit: float, door_reach: float, window_reach: float, cfg: Settings) -> bool:
    changed = False
    for opening in seeds:
        reach = window_reach if opening.kind == "window" else door_reach
        axis = np.array([math.cos(opening.axis_angle), math.sin(opening.axis_angle)])
        centre = np.asarray(opening.center, float)
        half = float(opening.width_m) * upm / 2.0
        covered = False
        candidates: list[tuple[float, int, float]] = []
        for index, line in enumerate(result):
            if abs(float(line.d[0] * axis[1] - line.d[1] * axis[0])) >= sin_limit:
                continue
            relative = centre - line.a
            if abs(float(relative @ line.n)) >= cfg.merge_offset_m * upm:
                continue
            u = float(relative @ line.d)
            if -half * 0.5 <= u <= line.length + half * 0.5:
                covered = True
                break
            gap = -u if u < 0 else u - line.length
            slack = 0.10 * upm if opening.kind == "window" else 1e-9  # the window edge and the snapped wall end differ by a few cm
            if gap <= half + slack and gap + half <= reach:
                candidates.append((gap, index, u))
        if covered or not candidates:
            continue
        _, index, u = min(candidates)
        line = result[index]
        changed = True
        if u < 0:
            extra = -(u - half)
            result[index] = replace(line, a=line.a + line.d * (u - half), length=line.length + extra, length_m=(line.length + extra) / upm, pad_start_m=line.pad_start_m + extra / upm)
        else:
            extra = (u + half) - line.length
            result[index] = replace(line, length=line.length + extra, length_m=(line.length + extra) / upm, pad_end_m=line.pad_end_m + extra / upm)
    return changed


def wall_faces(lines: list[WallLine]) -> list[WallFace]:
    return [WallFace(line, side) for line in lines for side in (1, -1)]


def grid_shape(line: WallLine, plan: BuildingPlan, cell_m: float) -> tuple[int, int]:
    """(nv, nu) of the wall-plane grid."""
    nu = int(round(line.length_m / cell_m))
    nv = int(round(storey_height_m(plan) / cell_m))
    return nv, nu


def face_grid_points(face: WallFace, plan: BuildingPlan, cell_m: float, *, row_step: int = 1) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Cell centres on the face plane. Returns (u_m (nu), v_m (nv), local points (nv, nu, 3))."""
    upm = plan.units_per_metre
    line = face.line
    nv, nu = grid_shape(line, plan, cell_m)
    u = (np.arange(nu) + 0.5) * cell_m
    v = (np.arange(nv) + 0.5) * cell_m
    xy = line.a[None, :] + line.d[None, :] * (u[:, None] * upm) + face.n[None, :] * (line.thickness / 2.0)
    points = np.zeros((nv, nu, 3))
    points[:, :, 0] = xy[None, :, 0]
    points[:, :, 1] = xy[None, :, 1]
    points[:, :, 2] = plan.floor_z + v[:, None] * upm
    if row_step > 1:
        points = points[::row_step]
        v = v[::row_step]
    return u, v, points


def host_piece(line: WallLine, u_m: float, plan: BuildingPlan) -> tuple[PlanWall, np.ndarray, np.ndarray]:
    """Plan wall piece covering (or nearest to) position ``u_m`` on the line."""
    upm = plan.units_per_metre
    by_id = {wall.id: wall for wall in plan.walls}
    best: tuple[float, PlanWall, np.ndarray, np.ndarray] | None = None
    for piece_id in line.pieces:
        wall = by_id[piece_id]
        a, b = np.asarray(wall.centreline[0], float), np.asarray(wall.centreline[1], float)
        ua, ub = float((a - line.a) @ line.d) / upm, float((b - line.a) @ line.d) / upm
        lo, hi = min(ua, ub), max(ua, ub)
        distance = 0.0 if lo <= u_m <= hi else min(abs(u_m - lo), abs(u_m - hi))
        if best is None or distance < best[0]:
            best = (distance, wall, a, b)
    assert best is not None
    return best[1], best[2], best[3]


def opening_placement(line: WallLine, u_m: float, plan: BuildingPlan, host_id: str | None = None) -> tuple[list[float], str, float, float]:
    """(centre in plan units on the host piece's centreline, host wall id, axis angle in [0, pi), thickness_m).

    ``host_id`` keeps a given wall piece of the line as host (an opening trimmed along its wall stays on the wall it was measured on,
    also where another collinear piece with a small offset lies nearer to the new centre).
    """
    upm = plan.units_per_metre
    wall, a, b = host_piece(line, u_m, plan)
    if host_id is not None and host_id != wall.id and host_id in line.pieces:
        wall = next(item for item in plan.walls if item.id == host_id)
        a, b = np.asarray(wall.centreline[0], float), np.asarray(wall.centreline[1], float)
    position = line.a + line.d * u_m * upm
    along = b - a
    norm2 = float(along @ along)
    centre = a + along * float(np.clip((position - a) @ along / norm2, -1e9, 1e9)) if norm2 > 0 else position
    angle = math.atan2(along[1], along[0]) % math.pi
    return [round(float(centre[0]), 3), round(float(centre[1]), 3)], wall.id, angle, float(wall.thickness_m)


def walls_fingerprint(lines: list[WallLine]) -> list[dict[str, Any]]:
    """Wall set description used for freshness (end points, rounded to mm)."""
    entries = []
    for line in lines:
        p0 = line.a
        p1 = line.a + line.d * line.length
        entries.append({"p0": [round(float(p0[0]), 3), round(float(p0[1]), 3)], "p1": [round(float(p1[0]), 3), round(float(p1[1]), 3)], "t": round(float(line.thickness), 3)})
    entries.sort(key=lambda item: (item["p0"][0], item["p0"][1], item["p1"][0], item["p1"][1]))
    return entries


def walls_match(first: list[dict[str, Any]], second: list[dict[str, Any]], tolerance: float) -> bool:
    """True when both wall sets have the same lines within ``tolerance`` (plan units) at both end points."""
    if len(first) != len(second):
        return False
    remaining = list(second)
    for item in first:
        found = None
        for index, other in enumerate(remaining):
            if all(math.dist(item[key], other[key]) <= tolerance for key in ("p0", "p1")):
                found = index
                break
        if found is None:
            return False
        remaining.pop(found)
    return True
