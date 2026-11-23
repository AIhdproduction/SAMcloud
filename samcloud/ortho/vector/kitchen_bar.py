"""Raised bar top of a free-standing kitchen leg (peninsula), measured in the scan.

A bar top is a board above the worktop level, carried by the body or the back panel of the peninsula and usually
overhanging it on the room side.  It is established only from what the scan shows:

* a level above the worktop (``above_m`` .. ``max_level_m``) where most points beside and above the leg lie;
* along the leg, a strip whose outer edge shows the top at that level over a good share of the leg's length (a stool or
  the back of a chair standing at the bar shows it over its own width only);
* from that edge towards the worktop the strip goes on while the top, or things standing on it, are seen and no worktop
  surface is (things on a bar hide the board from above; where the worktop is seen there is no bar);
* it lies over the body of the leg: a top standing clear of the leg is another piece of furniture;
* it does not cover the whole leg: a leg raised over its whole depth is a counter at that level, not a bar on a counter.

The underside is read at the free edge of the board (its edge face); without one the board has the worktop thickness.
Nothing is completed by a default depth or by symmetry.  Coordinates are wall-frame plan units, heights are metres.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np

from .kitchen_fronts import ScanPoints, _runs


@dataclass(frozen=True, slots=True)
class BarTopSettings:
    """Evidence thresholds of a raised bar top."""

    enabled: bool = True
    # The level: at least a hand's width above the worktop (below that it is the upstand of the worktop) and at most standing-bar
    # height.  Same values as the surface-model bar rule of the kitchen layout (bar_above_m, bar_max_m).
    above_m: float = 0.08
    max_level_m: float = 1.30
    # Scan noise of a horizontal board (about 1 cm) plus the 1 cm level bins.
    level_tolerance_m: float = 0.03
    # Bottles, a toaster, a knife block standing on the bar: up to this far above it.
    object_height_m: float = 0.45
    # Searched this far beside the leg: no bar top overhangs its body by more (seat depth of a breakfast bar is 0.25-0.40 m).
    reach_m: float = 0.60
    station_m: float = 0.04  # along the leg: the stations of the cabinet fronts
    cell_m: float = 0.02  # across the leg
    # The free edge of the board is seen at its level along at least this share of the leg.  A stool or a chair back at the bar
    # is 0.35-0.45 m wide: under 0.40 of any leg of 1.2 m and more (two seats).
    edge_share: float = 0.40
    # Inside the strip the top or what stands on it is seen at least this often; black stone and objects leave gaps
    # (measured on four scans of one kitchen: 0.29-0.81 inside the bar, 0.04-0.21 in the cells beside it).
    inside_share: float = 0.25
    min_depth_m: float = 0.15  # narrower is a rail or an upstand, not a top (bar_min_depth_m of the kitchen layout)
    max_depth_m: float = 0.90  # deeper is a table
    min_carry_m: float = 0.05  # it lies at least this far over the body: the thickness of a back panel with its cover
    max_cover_share: float = 0.80  # of the leg depth; more is the counter itself at that level
    min_length_share: float = 0.50  # of the leg (bar_min_length_share of the kitchen layout)
    min_length_m: float = 0.60  # one seat
    gap_m: float = 0.40  # things standing on the bar interrupt it along the leg
    end_snap_m: float = 0.15  # a bar that nearly reaches an end of the leg reaches it (as the surface-model rule does)
    thickness_m: float = 0.04  # a board whose edge face is not seen: the worktop thickness of the reconstruction
    worktop_search_m: float = 0.08  # the scanned worktop level is looked for this far around the surface-model level
    min_points: int = 60


@dataclass(frozen=True, slots=True)
class BarTop:
    """A measured bar top: wall-frame box, top level and underside (metres above the floor), what carries it."""

    u0: float
    u1: float
    v0: float
    v1: float
    level_m: float
    bottom_m: float
    worktop_m: float  # the worktop of the leg as scanned
    # the part lying over the leg (wall-frame box), where the board stands on the body; None when it only touches the back
    support: tuple[float, float, float, float] | None
    # the long edges of the board that lie inside the leg: the step from worktop to bar (wall-frame segments)
    steps: tuple[tuple[tuple[float, float], tuple[float, float]], ...]
    evidence: dict[str, Any]


def measure_bar_top(leg: Any, scan: ScanPoints | None, counter_top_m: float, units_per_metre: float, settings: BarTopSettings = BarTopSettings()) -> BarTop | None:
    """The bar top of the free-standing ``leg`` (``_LegBox``), or None without the full evidence (see the module docstring)."""
    upm = max(float(units_per_metre), 1e-9)
    if not settings.enabled or scan is None or leg.back is not None or len(scan.uv) < settings.min_points:
        return None
    horizontal = bool(leg.horizontal)
    along, across = (scan.uv[:, 0], scan.uv[:, 1]) if horizontal else (scan.uv[:, 1], scan.uv[:, 0])
    a0, a1, c0, c1 = (leg.u0, leg.u1, leg.v0, leg.v1) if horizontal else (leg.v0, leg.v1, leg.u0, leg.u1)
    reach, step, cell = settings.reach_m * upm, settings.station_m * upm, settings.cell_m * upm
    z = scan.z
    zone = (along >= a0) & (along <= a1) & (across >= c0 - reach) & (across <= c1 + reach)
    # the worktop of this leg as the scan shows it (the level handed in is the median of the surface model, objects included)
    on_leg = zone & (across >= c0) & (across <= c1) & (np.abs(z - counter_top_m) <= settings.worktop_search_m)
    if int(on_leg.sum()) >= settings.min_points:
        worktop_edges = np.arange(counter_top_m - settings.worktop_search_m, counter_top_m + settings.worktop_search_m + 0.0101, 0.01)
        worktop_histogram, _ = np.histogram(z[on_leg], bins=worktop_edges)
        counter_top_m = float(worktop_edges[int(np.argmax(np.convolve(worktop_histogram, np.ones(3), mode="same")))] + 0.005)
    low, high = counter_top_m + settings.above_m, settings.max_level_m
    raised = zone & (z >= low) & (z <= high)
    if high - low < 0.05 or int(raised.sum()) < settings.min_points:
        return None

    # the level where the raised points gather
    edges = np.arange(low, high + 0.0101, 0.01)
    histogram, _ = np.histogram(z[raised], bins=edges)
    level = float(edges[int(np.argmax(np.convolve(histogram, np.ones(3), mode="same")))] + 0.005)
    tolerance = settings.level_tolerance_m

    stations = max(1, int(math.ceil((a1 - a0) / step)))
    origin = c0 - reach
    cells = max(1, int(math.ceil((c1 - c0 + 2.0 * reach) / cell)))

    def seen(mask: np.ndarray) -> np.ndarray:
        """(cells, stations): where points of ``mask`` lie."""
        grid = np.zeros((cells, stations), dtype=bool)
        rows = np.clip(((across[mask] - origin) / cell).astype(np.int64), 0, cells - 1)
        cols = np.clip(((along[mask] - a0) / step).astype(np.int64), 0, stations - 1)
        grid[rows, cols] = True
        return grid

    top = seen(zone & (np.abs(z - level) <= tolerance))
    on = seen(zone & (z >= level - tolerance) & (z <= level + settings.object_height_m))
    worktop = seen(zone & (np.abs(z - counter_top_m) <= tolerance))
    top_share, on_share, worktop_share = top.mean(axis=1), on.mean(axis=1), worktop.mean(axis=1)
    centres = origin + (np.arange(cells) + 0.5) * cell
    over_leg = (centres >= c0) & (centres <= c1)

    seed = int(np.argmax(top_share))
    if top_share[seed] < settings.edge_share:
        return None
    # the strip: where the top itself is seen; or where only things at and above its level are seen and no worktop surface is
    covered = (top_share >= settings.inside_share) | ((on_share >= settings.inside_share) & ~(over_leg & (worktop_share >= settings.inside_share)))
    if not covered[seed]:
        return None
    first = last = seed
    while first > 0 and covered[first - 1]:
        first -= 1
    while last < cells - 1 and covered[last + 1]:
        last += 1
    # an end beside the leg is the free edge of the board: the top itself is seen there (a chair standing at the bar is not)
    while first < seed and not over_leg[first] and top_share[first] < settings.edge_share:
        first += 1
    while last > seed and not over_leg[last] and top_share[last] < settings.edge_share:
        last -= 1
    e0, e1 = origin + first * cell, origin + (last + 1) * cell
    depth = e1 - e0
    if depth < settings.min_depth_m * upm or depth > settings.max_depth_m * upm:
        return None
    over = min(e1, c1) - max(e0, c0)
    if over < settings.min_carry_m * upm or over > settings.max_cover_share * (c1 - c0):
        return None

    # along the leg: the stretch that shows the strip, objects on it bridged
    flags = on[first:last + 1].any(axis=0)
    runs = _runs(flags, max(1, int(round(settings.gap_m * upm / step))))
    if not runs:
        return None
    start, stop = max(runs, key=lambda run: run[1] - run[0])
    b0, b1 = a0 + start * step, min(a1, a0 + stop * step)
    if b0 - a0 <= settings.end_snap_m * upm:
        b0 = a0
    if a1 - b1 <= settings.end_snap_m * upm:
        b1 = a1
    length = b1 - b0
    if length < settings.min_length_m * upm or length < settings.min_length_share * (a1 - a0):
        return None

    # the board itself: its level is read again on the strip alone (5 mm), away from what stands on the worktop beside it
    on_strip = (along >= b0) & (along <= b1) & (across >= e0) & (across <= e1) & (np.abs(z - level) <= 2.0 * tolerance)
    if int(on_strip.sum()) >= settings.min_points:
        fine_edges = np.arange(level - 2.0 * tolerance, level + 2.0 * tolerance + 0.0051, 0.005)
        fine, _ = np.histogram(z[on_strip], bins=fine_edges)
        level = float(fine_edges[int(np.argmax(np.convolve(fine, np.ones(3), mode="same")))] + 0.0025)

    # the underside, read on the edge face of a free edge
    bottoms: list[float] = []
    for edge, free in ((e0, e0 < c0), (e1, e1 > c1)):
        if not free:
            continue
        face = (along >= b0) & (along <= b1) & (np.abs(across - edge) <= tolerance * upm) & (z <= level + tolerance) & (z >= counter_top_m - 0.02)
        if int(face.sum()) < 8:
            continue
        # 2 cm layers downwards; layer 0 is the top of the board itself (level - 2 cm and above)
        layers = max(2, int(math.ceil((level - counter_top_m + 0.02) / 0.02)))
        grid = np.zeros((layers, stations), dtype=bool)
        rows = np.clip(np.floor((level - z[face]) / 0.02).astype(np.int64), 0, layers - 1)
        grid[rows, np.clip(((along[face] - a0) / step).astype(np.int64), 0, stations - 1)] = True
        share = grid[:, start:stop].mean(axis=1) if stop > start else grid.mean(axis=1)
        reached = 0
        while reached < layers and share[reached] >= settings.inside_share:
            reached += 1
        if reached >= 2:  # more than the top layer: the edge face of the board
            bottoms.append(level - reached * 0.02)
    bottom = min(bottoms) if bottoms else level - settings.thickness_m
    bottom = float(min(max(bottom, counter_top_m), level - 0.02))

    support = None
    if bottom > counter_top_m + 0.005:
        support = (b0, max(e0, c0), b1, min(e1, c1)) if horizontal else (max(e0, c0), b0, min(e1, c1), b1)
    steps = tuple(
        (((b0, edge), (b1, edge)) if horizontal else ((edge, b0), (edge, b1)))
        for edge in (e0, e1) if c0 + 1e-9 < edge < c1 - 1e-9
    )
    u0, u1, v0, v1 = (b0, b1, e0, e1) if horizontal else (e0, e1, b0, b1)
    evidence = {
        "level_m": round(level, 3), "bottom_m": round(bottom, 3), "worktop_m": round(counter_top_m, 3), "depth_m": round(depth / upm, 3), "length_m": round(length / upm, 3),
        "over_body_m": round(over / upm, 3), "edge_share": round(float(top_share[seed]), 3),
        "inside_share": round(float(on_share[first:last + 1].min()), 3), "edge_face": bool(bottoms),
    }
    return BarTop(float(u0), float(u1), float(v0), float(v1), level, bottom, float(counter_top_m), support, steps, evidence)


__all__ = ["BarTop", "BarTopSettings", "measure_bar_top"]
