"""Compare ground filters (PMF vs CSF) on one classified cloud.

Metrics per method (see ``compare_ground_filters``):

* ``footprint_share``  share of ground points inside building/tree footprints (lower = better)
* ``plane_rms_m``      RMS residual of road/sidewalk ground points to local planes (lower = better)
* ``profile_error_m``  max height error of the ground profile vs. the raw lowest-point profile
* ``coverage``         share of occupied 1 m cells that contain a ground point (higher = better)
"""
from __future__ import annotations

import logging
import math

import numpy as np
import shapely
from scipy import ndimage

from samcloud.drone.contracts import (
    BUILDING, PATH, ROAD, ROOF_FLAT, ROOF_INCLINED, SEALED_TERRAIN, SIDEWALK, TREE, PointSet,
)

logger = logging.getLogger(__name__)

_OBJECT_CLASSES = (BUILDING, ROOF_INCLINED, ROOF_FLAT, TREE)
_PAVEMENT_CLASSES = (ROAD, SIDEWALK, PATH, SEALED_TERRAIN)
_PLANE_CELL_M = 1.0
_PLANE_MIN_POINTS = 6
_OCCUPANCY_CELL_M = 0.5
_STATION_STEP_M = 0.25
_CORRIDOR_M = 0.25

# pick_best scoring: weighted mean of normalised penalties (each in 0..1, lower = better).
# Metrics that are NaN for every method are skipped and the weights renormalised.
SCORE_WEIGHTS = {"footprint_share": 4.0, "plane_rms_m": 2.0, "profile_error_m": 2.0, "coverage": 1.0}
_RMS_SCALE_M = 0.10        # plane RMS of 10 cm (or worse) = full penalty
_PROFILE_SCALE_M = 0.30    # profile error of 30 cm (or worse) = full penalty


def _footprint_cells(ps: PointSet) -> tuple[np.ndarray, np.ndarray]:
    """2-D occupancy raster of building/tree points (closed, holes filled) and its origin."""
    obj = np.isin(ps.classes, _OBJECT_CLASSES)
    xy = np.asarray(ps.xyz[:, :2], dtype=np.float64)
    origin = xy.min(axis=0)
    shape_xy = np.floor((xy.max(axis=0) - origin) / _OCCUPANCY_CELL_M).astype(int) + 1
    grid = np.zeros((shape_xy[0], shape_xy[1]), dtype=bool)
    if obj.any():
        ij = np.floor((xy[obj] - origin) / _OCCUPANCY_CELL_M).astype(int)
        grid[ij[:, 0], ij[:, 1]] = True
        grid = ndimage.binary_closing(grid, structure=np.ones((3, 3)), iterations=1)
        grid = ndimage.binary_fill_holes(grid)
    return grid, origin


def _share_in_footprints(ps: PointSet, ground_xy: np.ndarray, footprints) -> float:
    if len(ground_xy) == 0:
        return math.nan
    if footprints is not None:
        inside = np.zeros(len(ground_xy), dtype=bool)
        for poly in footprints:
            inside |= shapely.contains_xy(poly, ground_xy[:, 0], ground_xy[:, 1])
        return float(inside.mean())
    grid, origin = _footprint_cells(ps)
    ij = np.floor((ground_xy - origin) / _OCCUPANCY_CELL_M).astype(int)
    ij = np.clip(ij, 0, np.array(grid.shape) - 1)
    return float(grid[ij[:, 0], ij[:, 1]].mean())


def _plane_rms(xyz: np.ndarray) -> float:
    """RMS residual to least-squares planes fitted per 1 m cell (cells with >= 6 points)."""
    if len(xyz) == 0:
        return math.nan
    ij = np.floor(xyz[:, :2] / _PLANE_CELL_M).astype(np.int64)
    _, inverse, counts = np.unique(ij, axis=0, return_inverse=True, return_counts=True)
    inverse = inverse.ravel()
    centre = (ij[:, :2] + 0.5) * _PLANE_CELL_M
    dx = xyz[:, 0] - centre[:, 0]
    dy = xyz[:, 1] - centre[:, 1]
    z = xyz[:, 2]
    m = len(counts)

    def s(values):
        return np.bincount(inverse, weights=values, minlength=m)

    normal = np.zeros((m, 3, 3))
    normal[:, 0, 0], normal[:, 0, 1], normal[:, 0, 2] = s(dx * dx), s(dx * dy), s(dx)
    normal[:, 1, 0], normal[:, 1, 1], normal[:, 1, 2] = normal[:, 0, 1], s(dy * dy), s(dy)
    normal[:, 2, 0], normal[:, 2, 1], normal[:, 2, 2] = normal[:, 0, 2], normal[:, 1, 2], counts
    rhs = np.stack([s(dx * z), s(dy * z), s(z)], axis=1)
    good = counts >= _PLANE_MIN_POINTS
    if not good.any():
        return math.nan
    coef = np.zeros((m, 3))
    try:
        coef[good] = np.linalg.solve(normal[good] + 1e-9 * np.eye(3), rhs[good][..., None])[..., 0]
    except np.linalg.LinAlgError:
        coef[good] = np.stack([np.linalg.lstsq(a, b, rcond=None)[0] for a, b in zip(normal[good], rhs[good])])
    use = good[inverse]
    pred = coef[inverse, 0] * dx + coef[inverse, 1] * dy + coef[inverse, 2]
    return float(np.sqrt(np.mean((z[use] - pred[use]) ** 2)))


def _profile_lowest(points: np.ndarray, line, n_stations: int) -> np.ndarray:
    """Lowest Z per station along ``line`` within the corridor (NaN where no point)."""
    low = np.full(n_stations, np.nan)
    if len(points) == 0:
        return low
    pts = shapely.points(points[:, 0], points[:, 1])
    near = shapely.distance(line, pts) <= _CORRIDOR_M
    if not near.any():
        return low
    along = shapely.line_locate_point(line, pts[near])
    station = np.clip(np.rint(along / _STATION_STEP_M).astype(int), 0, n_stations - 1)
    buf = np.full(n_stations, np.inf)
    np.minimum.at(buf, station, points[near, 2])
    low[np.isfinite(buf)] = buf[np.isfinite(buf)]
    return low


def _profile_metrics(ps: PointSet, ground_xyz: np.ndarray, profiles) -> tuple[float, float]:
    """(max |ground - raw lowest| over all stations, share of stations with ground)."""
    worst = math.nan
    covered = 0
    total = 0
    xyz = np.asarray(ps.xyz, dtype=np.float64)
    for prof in profiles:
        line = prof if hasattr(prof, "length") else shapely.linestrings(np.asarray(prof, dtype=np.float64)[:, :2])
        n = int(max(2, math.ceil(line.length / _STATION_STEP_M) + 1))
        raw = _profile_lowest(xyz, line, n)
        got = _profile_lowest(ground_xyz, line, n)
        both = np.isfinite(raw) & np.isfinite(got)
        total += int(np.isfinite(raw).sum())
        covered += int(both.sum())
        if both.any():
            err = float(np.abs(got[both] - raw[both]).max())
            worst = err if math.isnan(worst) else max(worst, err)
    return worst, (covered / total if total else math.nan)


def _coverage(ps: PointSet, ground_xy: np.ndarray) -> float:
    xy = np.asarray(ps.xyz[:, :2], dtype=np.float64)
    cell = 1.0
    occupied = np.unique(np.floor(xy / cell).astype(np.int64), axis=0)
    if len(ground_xy) == 0:
        return 0.0
    got = np.unique(np.floor(ground_xy / cell).astype(np.int64), axis=0)
    key = lambda a: a[:, 0] * 2_000_003 + a[:, 1]  # noqa: E731
    return float(np.isin(key(got), key(occupied)).sum() / len(occupied))


def compare_ground_filters(
    ps: PointSet, masks: dict[str, np.ndarray], *, footprints=None, profiles=None,
) -> dict:
    """Metrics per ground filter: ``{name: {...}}`` (see module docstring).

    ``footprints``: optional shapely polygons (local XY); default = 2-D occupancy of building
    and tree classes.  ``profiles``: optional polylines (``(N,2)`` arrays or ``LineString``)
    along kerbs/slopes; the raw lowest-point profile of the whole cloud is the reference.
    """
    xyz = np.asarray(ps.xyz, dtype=np.float64)
    result: dict[str, dict] = {}
    for name, mask in masks.items():
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != (len(ps),):
            raise ValueError(f"mask {name!r} must have shape ({len(ps)},)")
        ground = xyz[mask]
        pavement = ground[np.isin(ps.classes[mask], _PAVEMENT_CLASSES)]
        entry = {
            "ground_points": int(mask.sum()),
            "ground_share": float(mask.mean()) if len(ps) else math.nan,
            "footprint_share": _share_in_footprints(ps, ground[:, :2], footprints),
            "plane_rms_m": _plane_rms(pavement),
            "coverage": _coverage(ps, ground[:, :2]),
            "profile_error_m": math.nan,
            "profile_coverage": math.nan,
        }
        if profiles:
            entry["profile_error_m"], entry["profile_coverage"] = _profile_metrics(ps, ground, profiles)
        result[name] = entry
    return result


def _penalties(entry: dict) -> dict[str, float]:
    pen: dict[str, float] = {}
    if math.isfinite(entry.get("footprint_share", math.nan)):
        pen["footprint_share"] = float(entry["footprint_share"])
    if math.isfinite(entry.get("plane_rms_m", math.nan)):
        pen["plane_rms_m"] = min(1.0, entry["plane_rms_m"] / _RMS_SCALE_M)
    if math.isfinite(entry.get("profile_error_m", math.nan)):
        pen["profile_error_m"] = min(1.0, entry["profile_error_m"] / _PROFILE_SCALE_M)
    pen["coverage"] = 1.0 - float(entry.get("coverage", 0.0))
    return pen


def pick_best(metrics: dict[str, dict]) -> tuple[str, str]:
    """Return ``(name, reason)`` of the best filter.

    Score = weighted mean of penalties in 0..1 (lower wins), weights ``SCORE_WEIGHTS``:
    footprint_share 4 (share itself), plane_rms_m 2 (rms / 10 cm, capped), profile_error_m 2
    (error / 30 cm, capped), coverage 1 (1 - coverage).  A metric that is NaN (e.g. no profiles
    given) is skipped for every method; ties go to the alphabetically first name.
    """
    if not metrics:
        raise ValueError("no metrics to choose from")
    pens = {name: _penalties(entry) for name, entry in metrics.items()}
    usable = set.intersection(*(set(p) for p in pens.values()))
    scores = {}
    for name, p in pens.items():
        weight = sum(SCORE_WEIGHTS[k] for k in usable)
        scores[name] = sum(SCORE_WEIGHTS[k] * p[k] for k in usable) / weight
    best = min(sorted(scores), key=lambda n: scores[n])
    parts = ", ".join(f"{k}={pens[best][k]:.3f}" for k in sorted(usable))
    others = ", ".join(f"{n}={s:.3f}" for n, s in sorted(scores.items()) if n != best)
    reason = f"lowest weighted penalty {scores[best]:.3f} ({parts})" + (f" vs {others}" if others else "")
    return best, reason
