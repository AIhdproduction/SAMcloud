"""Cloth simulation ground filter (CSF), after Zhang et al. (2016).

The point cloud is turned upside down, a cloth grid drops onto the inverted
surface under gravity, particles stop where they touch the inverted cloud and
the remaining particles are held by internal springs (rigidness).  The cloth
therefore bridges buildings and tree crowns and follows the terrain.

The implementation is vectorised NumPy: the cloud is reduced to one collision
height per cloth cell (binned maximum of the inverted Z), so 20 M points cost a
single binning pass plus one iteration loop over the cloth grid.

Limitation: the cloth is a membrane, so very wide (> ~40 m) low flat roofs
sag towards the roof; combine with the footprint comparison in ``compare``.
"""
from __future__ import annotations

import logging

import numpy as np
from scipy import ndimage

logger = logging.getLogger(__name__)

_GRAVITY = 0.05           # gravity acceleration (m per time_step^2 per iteration, finest level)
_CONVERGENCE_M = 1e-3      # early stop when the cloth moves less than this
_MAX_SLOPE_TAN = 1.7       # cap of the slope term (about 60 degrees)
_SNAP_FACTOR = 3.0         # slope post-processing snaps floating cells within 3 x class_threshold
_CHUNK = 4_000_000


def _bilinear(grid: np.ndarray, origin: np.ndarray, cell: float, xy: np.ndarray) -> np.ndarray:
    """Sample a cell-centred grid ``[y, x]`` bilinearly (clamped at the border)."""
    u = np.clip((xy[:, 0] - origin[0]) / cell - 0.5, 0, grid.shape[1] - 1)
    v = np.clip((xy[:, 1] - origin[1]) / cell - 0.5, 0, grid.shape[0] - 1)
    x0 = np.floor(u).astype(np.int64)
    y0 = np.floor(v).astype(np.int64)
    x1 = np.minimum(x0 + 1, grid.shape[1] - 1)
    y1 = np.minimum(y0 + 1, grid.shape[0] - 1)
    tx = u - x0
    ty = v - y0
    return ((1 - tx) * (1 - ty) * grid[y0, x0] + tx * (1 - ty) * grid[y0, x1]
            + (1 - tx) * ty * grid[y1, x0] + tx * ty * grid[y1, x1])


def _fill_nearest(values: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Replace invalid cells by the value of the nearest valid cell."""
    if valid.all():
        return values
    idx = ndimage.distance_transform_edt(~valid, return_distances=False, return_indices=True)
    return values[idx[0], idx[1]]


def _neighbour_sum(h: np.ndarray) -> np.ndarray:
    """Sum of the four neighbour differences (edge replicated, i.e. zero at the border)."""
    p = np.pad(h, 1, mode="edge")
    return (p[:-2, 1:-1] + p[2:, 1:-1] + p[1:-1, :-2] + p[1:-1, 2:]) - 4.0 * h


def _run_level(obstacle: np.ndarray, cloth: np.ndarray, *, passes: int, drop: float, iterations: int) -> np.ndarray:
    """Gravity + spring iterations on one grid level (float32, cloth >= obstacle)."""
    cloth = np.maximum(cloth, obstacle)
    movable = cloth > obstacle
    step = np.float32(drop)
    quarter = np.float32(0.25)
    for iteration in range(iterations):
        previous = cloth
        cloth = np.where(movable, cloth - step, cloth)
        cloth = np.maximum(cloth, obstacle)
        movable &= cloth > obstacle
        for _ in range(passes):
            cloth = np.where(movable, cloth + quarter * _neighbour_sum(cloth), cloth)
        cloth = np.maximum(cloth, obstacle)
        movable &= cloth > obstacle
        if float(np.abs(cloth - previous).max()) < _CONVERGENCE_M:
            logger.debug("CSF level converged after %d iterations", iteration + 1)
            break
    return cloth


def simulate_cloth(
    collision: np.ndarray, *, rigidness: int, iterations: int, time_step: float,
) -> np.ndarray:
    """Drop a cloth onto ``collision`` (inverted heights, ``[y, x]``) and return the resting cloth.

    A coarse-to-fine pyramid (factor 4 per level, gravity scaled with the square of the cell
    size so the physical sag stays the same) brings the cloth near its resting position
    quickly; the finest level then only refines it.
    """
    ny, nx = collision.shape
    levels = 0
    while min(ny, nx) / (4 ** (levels + 1)) >= 8 and levels < 4:
        levels += 1
    block = 4 ** levels
    pad_y, pad_x = (-ny) % block, (-nx) % block
    top = float(collision.max())
    obstacle = np.pad((collision - top).astype(np.float32), ((0, pad_y), (0, pad_x)), mode="edge")
    passes = max(1, int(rigidness)) * 2
    base_drop = _GRAVITY * time_step * time_step
    cloth = None
    for level in range(levels, -1, -1):
        factor = 4 ** level
        if factor == 1:
            pooled = obstacle
        else:
            h, w = obstacle.shape
            pooled = obstacle.reshape(h // factor, factor, w // factor, factor).max(axis=(1, 3))
        if cloth is None:
            cloth = np.zeros_like(pooled)
        else:
            cloth = ndimage.zoom(cloth, 4, order=1, mode="nearest", grid_mode=True).astype(np.float32)
        cloth = _run_level(pooled, cloth, passes=passes, drop=base_drop * factor * factor, iterations=iterations)
    return cloth[:ny, :nx].astype(np.float64) + top


def cloth_simulation_filter(
    xyz: np.ndarray, *,
    cloth_resolution: float = 0.5,
    rigidness: int = 2,
    class_threshold: float = 0.15,
    iterations: int = 500,
    time_step: float = 0.65,
    slope_smooth: bool = True,
) -> np.ndarray:
    """Return a boolean ground mask (``True`` = ground) for ``xyz`` (N,3, Z up).

    ``class_threshold`` is the maximum point-to-cloth distance for ground.  With
    ``slope_smooth`` enables the slope post-processing: cells whose cloth hovers less than
    ``3 * class_threshold`` below the lowest point are snapped onto it (membrane rounding of
    terrain edges), and the threshold grows with the local cloth slope (by
    ``slope * cloth_resolution``) so steep embankments stay ground although the cloth only
    knows one lowest point per cell.
    """
    pts = np.asarray(xyz, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError("xyz must have shape (N, 3)")
    if cloth_resolution <= 0 or iterations < 1 or rigidness < 1:
        raise ValueError("cloth_resolution > 0, iterations >= 1 and rigidness >= 1 are required")
    n = len(pts)
    if n == 0:
        return np.zeros(0, dtype=bool)
    finite = np.isfinite(pts).all(axis=1)
    if not finite.all():
        out = np.zeros(n, dtype=bool)
        out[finite] = cloth_simulation_filter(
            pts[finite], cloth_resolution=cloth_resolution, rigidness=rigidness,
            class_threshold=class_threshold, iterations=iterations, time_step=time_step,
            slope_smooth=slope_smooth)
        return out

    lo = pts[:, :2].min(axis=0)
    hi = pts[:, :2].max(axis=0)
    origin = lo - cloth_resolution
    nx = int(np.floor((hi[0] - origin[0]) / cloth_resolution)) + 2
    ny = int(np.floor((hi[1] - origin[1]) / cloth_resolution)) + 2
    zmin = np.full(nx * ny, np.inf)
    for start in range(0, n, _CHUNK):
        chunk = pts[start:start + _CHUNK]
        ix = np.floor((chunk[:, 0] - origin[0]) / cloth_resolution).astype(np.int64)
        iy = np.floor((chunk[:, 1] - origin[1]) / cloth_resolution).astype(np.int64)
        np.minimum.at(zmin, iy * nx + ix, chunk[:, 2])
    zmin = zmin.reshape(ny, nx)
    valid = np.isfinite(zmin)
    zmin = _fill_nearest(zmin, valid)

    inverted = simulate_cloth(-zmin, rigidness=rigidness, iterations=iterations, time_step=time_step)
    cloth_z = -inverted                                         # real-space cloth height
    if slope_smooth:
        # Slope post-processing (as in CSF): the membrane cuts convex terrain edges by a few
        # decimetres; floating cells that lie only slightly below their lowest point are
        # snapped onto it.  Bridged objects hover far higher and stay untouched.
        gap = zmin - cloth_z
        snap = (gap > 0) & (gap <= _SNAP_FACTOR * class_threshold)
        cloth_z = np.where(snap, zmin, cloth_z)
    allowance = np.zeros_like(cloth_z)
    if slope_smooth:
        gy, gx = np.gradient(cloth_z, cloth_resolution)
        allowance = np.minimum(np.hypot(gx, gy), _MAX_SLOPE_TAN) * cloth_resolution

    ground = np.zeros(n, dtype=bool)
    for start in range(0, n, _CHUNK):
        chunk = pts[start:start + _CHUNK]
        xy = chunk[:, :2]
        diff = chunk[:, 2] - _bilinear(cloth_z, origin, cloth_resolution, xy)
        tolerance = class_threshold + (_bilinear(allowance, origin, cloth_resolution, xy) if slope_smooth else 0.0)
        ground[start:start + _CHUNK] = np.abs(diff) <= tolerance
    return ground
