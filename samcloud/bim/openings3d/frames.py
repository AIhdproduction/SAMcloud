"""Door frame, head and headroom measured on the wall-plane grids (door-class points and all points in the wall band).

The door-class points (Cyclone 103) are the usable signal for the casing of a door: the two jamb strips at the
opening edges and the head strip above it.  The head height is the lowest height above ``frames.head_search_from_m``
where at least ``frames.head_row_fraction`` of the columns inside the opening have points of any class; the
headroom is the ceiling minus that head.  A door stands under a lintel (headroom of a few decimetres), a
passage reaches the ceiling (headroom of a few centimetres).  Every number is a plain measurement; the
classification rules live in :mod:`.classify`.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from .config import Settings


@dataclass
class FrameFeatures:
    """Measurements of one opening (``left`` / ``right`` are along the direction of its wall line, +u)."""

    ceiling_m: float
    head_m: float | None  # measured head (None: no row above the search height was occupied)
    headroom_m: float | None
    head_used_m: float  # head used for the strips (measured, else the default door head)
    jamb_cov: dict[str, float] = field(default_factory=dict)  # door-class row coverage of the jamb strip
    jamb_out_m: dict[str, float] = field(default_factory=dict)  # width of the door-class run outward of the jamb
    frame_jambs: list[str] = field(default_factory=list)  # jambs that are frame strips (side names)
    flat_leaf_sides: list[str] = field(default_factory=list)  # wide door-class run: a flat leaf, not a frame
    head_cov: float = 0.0
    head_peak_m: float | None = None
    head_frame: bool = False
    door_fraction: float = 0.0  # share of the opening cells (door rows) holding door-class points
    free_fraction: float = 0.0  # share of the opening cells without any point of any class
    blocked_fraction: float = 0.0  # share of the opening cells holding non-structural (furniture) points
    wall_fraction: float = 0.0  # share of the opening cells holding wall-class points
    side_wall_cov: dict[str, float] = field(default_factory=dict)  # wall-class row coverage of the wall next to each edge

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _rows(lo_m: float, hi_m: float, cell: float, rows: int) -> slice:
    return slice(max(int(round(lo_m / cell)), 0), min(max(int(round(hi_m / cell)), 0), rows))


def _cols(lo_m: float, hi_m: float, cell: float, columns: int) -> slice:
    return slice(min(max(int(round(lo_m / cell)), 0), columns), min(max(int(round(hi_m / cell)), 0), columns))


def _row_coverage(grid: np.ndarray, columns: slice, rows: slice, min_points: int) -> float:
    """Share of the grid rows whose points summed over ``columns`` reach ``min_points``."""
    sub = grid[rows, columns]
    if sub.shape[0] == 0 or sub.shape[1] == 0:
        return 0.0
    return float((sub.sum(1) >= min_points).mean())


def _column_coverage(grid: np.ndarray, columns: slice, rows: slice, min_points: int) -> float:
    sub = grid[rows, columns]
    if sub.shape[0] == 0 or sub.shape[1] == 0:
        return 0.0
    return float((sub.sum(0) >= min_points).mean())


def measure_head(any_grid: np.ndarray, u0: float, u1: float, config: Settings) -> float | None:
    """Lowest height (m) above the search height where enough opening columns hold points of any class."""
    cfg = config.frames
    cell = float(config.grid.cell_m)
    inner = (u1 - u0) * float(cfg.inner_fraction) / 2.0
    centre = (u0 + u1) / 2.0
    columns = _cols(centre - inner, centre + inner, cell, any_grid.shape[1])
    if columns.stop - columns.start < 1:
        return None
    start = int(round(float(cfg.head_search_from_m) / cell))
    for row in range(start, any_grid.shape[0]):
        if float((any_grid[row, columns] >= int(cfg.head_min_points)).mean()) >= float(cfg.head_row_fraction):
            return row * cell
    return None


def _jamb_columns(edge: float, side: int, cfg: Settings, cell: float, columns: int) -> slice:
    inside, outside = edge - side * float(cfg.jamb_inside_m), edge + side * float(cfg.jamb_outside_m)
    return _cols(min(inside, outside), max(inside, outside), cell, columns)


def _outward_run(door: np.ndarray, edge: float, side: int, cfg: Settings, cell: float) -> float:
    """Width (m) of the contiguous door-class run that starts ``jamb_inside_m`` inside the edge and runs outwards."""
    rows = _rows(float(cfg.out_rows_m[0]), float(cfg.out_rows_m[1]), cell, door.shape[0])
    run = 0
    for step in range(int(round(float(cfg.out_max_m) / cell))):
        start = edge - side * float(cfg.jamb_inside_m) + side * step * cell
        lo, hi = (start, start + cell) if side > 0 else (start - cell, start)
        coverage = _row_coverage(door, _cols(lo, hi, cell, door.shape[1]), rows, int(cfg.out_min_points))
        if coverage >= float(cfg.out_run_cov):
            run += 1
        elif run > 0 and coverage >= float(cfg.out_run_cov_continue):
            run += 1
        else:
            break
    return run * cell


def measure_opening(grids: dict[str, np.ndarray], u0: float, u1: float, ceiling_m: float, config: Settings, shared_edges: tuple[str, ...] = ()) -> FrameFeatures:
    """Frame, head and headroom features of the opening spanning ``u0..u1`` metres of a wall-line grid.

    ``shared_edges`` (``left`` / ``right``) names edges that another opening of the same wall line shares: the casing there
    belongs to that opening, so no jamb is counted for this one on that side.
    """
    cfg = config.frames
    cell = float(config.grid.cell_m)
    door, anyg = grids["door"], grids["any"]
    nv, nu = door.shape
    inner = (u1 - u0) * float(cfg.inner_fraction) / 2.0
    centre = (u0 + u1) / 2.0
    inner_cols = _cols(centre - inner, centre + inner, cell, nu)
    cells = (_rows(float(cfg.cell_rows_m[0]), float(cfg.cell_rows_m[1]), cell, nv), inner_cols)
    door_cells = grids["door"][cells]
    door_fraction = float((door_cells >= int(cfg.door_cell_points)).mean()) if door_cells.size else 0.0
    # A closed leaf fills the opening with door-class points; the head is then the lowest non-door point above it.
    head_grid = anyg - door if door_fraction >= float(cfg.closed_leaf_fraction) else anyg
    head = measure_head(head_grid, u0, u1, config)
    head_used = head if head is not None else float(config.doors.default_head_m)
    features = FrameFeatures(ceiling_m=ceiling_m, head_m=head, headroom_m=None if head is None else max(0.0, ceiling_m - head), head_used_m=head_used)
    body_rows = _rows(float(cfg.jamb_rows_from_m), head_used - float(cfg.jamb_rows_below_head_m), cell, nv)
    for name, edge, side in (("left", u0, -1), ("right", u1, 1)):
        columns = _jamb_columns(edge, side, cfg, cell, nu)
        coverage = _row_coverage(door, columns, body_rows, int(cfg.jamb_min_points))
        out = _outward_run(door, edge, side, cfg, cell)
        features.jamb_cov[name] = round(coverage, 3)
        features.jamb_out_m[name] = round(out, 3)
        if name in shared_edges:
            continue  # the jamb strip on a shared edge is the casing of the neighbouring opening
        if coverage >= float(cfg.jamb_coverage):
            (features.frame_jambs if out <= float(cfg.jamb_max_out_m) else features.flat_leaf_sides).append(name)
    head_rows = _rows(head_used - float(cfg.head_below_m), head_used + float(cfg.head_above_m), cell, nv)
    head_cols = _cols(u0 - float(cfg.head_margin_m), u1 + float(cfg.head_margin_m), cell, nu)
    features.head_cov = round(_column_coverage(door, head_cols, head_rows, int(cfg.head_min_points)), 3)
    peak_rows = _rows(float(cfg.head_search_from_m), ceiling_m, cell, nv)
    per_row = door[peak_rows, inner_cols].sum(1)
    if per_row.size and int(per_row.sum()) >= int(cfg.head_peak_min_points):
        features.head_peak_m = round(peak_rows.start * cell + int(np.argmax(per_row)) * cell, 3)
    features.head_frame = bool(features.head_cov >= float(cfg.head_coverage) and features.head_peak_m is not None and float(cfg.head_peak_range_m[0]) <= features.head_peak_m <= float(cfg.head_peak_range_m[1]))
    side_rows = _rows(float(cfg.side_rows_m[0]), float(cfg.side_rows_m[1]), cell, nv)
    for name, edge, side in (("left", u0, -1), ("right", u1, 1)):
        near, far = edge + side * float(cfg.side_from_m), edge + side * float(cfg.side_to_m)
        features.side_wall_cov[name] = round(_row_coverage(grids["wall"], _cols(min(near, far), max(near, far), cell, nu), side_rows, int(cfg.side_min_points)), 3)
    anyc, wallc, otherc = (grids[name][cells] for name in ("any", "wall", "other"))
    features.door_fraction = round(door_fraction, 3)
    if anyc.size:
        features.free_fraction = round(float((anyc <= int(cfg.free_max_points)).mean()), 3)
        features.wall_fraction = round(float((wallc >= int(cfg.wall_cell_points)).mean()), 3)
        features.blocked_fraction = round(float((otherc >= int(cfg.blocked_cell_points)).mean()), 3)
    return features
