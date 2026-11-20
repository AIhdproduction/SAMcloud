"""Measured tree and shrub instances from classified points and a DTM.

Every value is measured from the points.  A value the data cannot support is
``None`` (never guessed or defaulted).  ``dtm_sampler`` is a callable taking an
``(N, 2)`` array of local XY and returning ``(N,)`` terrain heights (NaN where
the terrain is unknown).
"""
from __future__ import annotations

import logging
import math
from typing import Callable

import numpy as np
from scipy import ndimage
from scipy.ndimage import gaussian_filter
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import Delaunay, cKDTree
from shapely import concave_hull
from shapely.geometry import MultiPoint, Polygon

from samcloud.drone.contracts import (
    MEDIUM_VEGETATION,
    SHRUB,
    TREE,
    TREE_TRUNK,
    Evidence,
    PointSet,
    QAEntry,
    ShrubInstance,
    TreeInstance,
    load_drone_config,
)
from samcloud.ortho.vector.raster_grid import RasterGrid
from samcloud.ortho.vector.trees import (
    detect_crowns,
    fit_trunk_diameter,
)

logger = logging.getLogger(__name__)

DtmSampler = Callable[[np.ndarray], np.ndarray]

_CROWN_DETECT_MIN_HEIGHT_M = 0.5   # detect low objects too, so they can be reported as rejected
_MIN_TRUNK_POINTS = 20
_TRUNK_BAND_M = (1.0, 1.6)
_TRUNK_FIT_MAX_RMS_M = 0.03
_SHRUB_EPS_M = 0.3
_SHRUB_MIN_POINTS = 10


def _vegetation_config(config: dict | None) -> dict:
    cfg = load_drone_config()["vegetation"] if config is None else config
    return cfg.get("vegetation", cfg)


def _sample(dtm_sampler: DtmSampler, xy: np.ndarray) -> np.ndarray:
    return np.asarray(dtm_sampler(np.asarray(xy, dtype=np.float64).reshape(-1, 2)), dtype=np.float64)


def canopy_height_model(
    ps: PointSet, dtm_sampler: DtmSampler, resolution: float = 0.25
) -> tuple[np.ndarray, RasterGrid | None]:
    """Max(Z - DTM) of tree-class points per cell; NaN where no tree point fell.

    Returns ``(chm, grid)``; with no tree points ``(empty (0,0) array, None)``.
    """
    mask = np.asarray(ps.classes) == TREE
    if not mask.any():
        return np.zeros((0, 0), dtype=np.float32), None
    xyz = np.asarray(ps.xyz, dtype=np.float64)[mask]
    ground = _sample(dtm_sampler, xyz[:, :2])
    ok = np.isfinite(ground)
    xyz, ground = xyz[ok], ground[ok]
    if len(xyz) == 0:
        return np.zeros((0, 0), dtype=np.float32), None
    pad = 2
    min_x = float(xyz[:, 0].min()) - pad * resolution
    max_y = float(xyz[:, 1].max()) + pad * resolution
    width = int(math.ceil((xyz[:, 0].max() - min_x) / resolution)) + pad + 1
    height = int(math.ceil((max_y - xyz[:, 1].min()) / resolution)) + pad + 1
    grid = RasterGrid(min_x, max_y, resolution, width, height)
    cols = np.clip(((xyz[:, 0] - min_x) / resolution).astype(int), 0, width - 1)
    rows = np.clip(((max_y - xyz[:, 1]) / resolution).astype(int), 0, height - 1)
    chm = np.full((height, width), -np.inf, dtype=np.float64)
    np.maximum.at(chm, (rows, cols), xyz[:, 2] - ground)
    chm[~np.isfinite(chm)] = np.nan
    return chm.astype(np.float32), grid


def _fill_chm(chm: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Close sampling holes inside canopies: returns (filled chm, canopy mask)."""
    observed = np.isfinite(chm)
    mask = ndimage.binary_closing(observed, structure=np.ones((3, 3), bool), iterations=2)
    mask = ndimage.binary_fill_holes(mask)
    value = np.where(observed, chm, 0.0)
    weight = observed.astype(np.float64)
    num = gaussian_filter(value, 1.5)
    den = gaussian_filter(weight, 1.5)
    interp = np.where(den > 1e-6, num / np.maximum(den, 1e-6), 0.0)
    filled = np.where(observed, chm, interp)
    filled = np.where(mask, filled, np.nan)
    return filled.astype(np.float32), mask


def _trunk_diameter(
    trunk_xyz: np.ndarray, trunk_xy: tuple[float, float], base_z: float
) -> tuple[float | None, float | None]:
    """Circle fit on the 1.0-1.6 m band; (diameter, rms) or (None, None)."""
    z = trunk_xyz[:, 2] - base_z
    band = trunk_xyz[(z >= _TRUNK_BAND_M[0]) & (z <= _TRUNK_BAND_M[1])][:, :2]
    if len(band) < 30:
        return None, None
    diameter = fit_trunk_diameter(band, trunk_xy)
    if diameter is None:
        return None, None
    a = np.column_stack([2 * band[:, 0], 2 * band[:, 1], np.ones(len(band))])
    b = (band ** 2).sum(axis=1)
    sol, *_ = np.linalg.lstsq(a, b, rcond=None)
    centre = sol[:2]
    resid = np.hypot(band[:, 0] - centre[0], band[:, 1] - centre[1]) - diameter / 2.0
    rms = float(np.sqrt(np.mean(resid ** 2)))
    if rms >= _TRUNK_FIT_MAX_RMS_M:
        return None, rms
    return float(diameter), rms


def detect_trees(
    ps: PointSet, dtm_sampler: DtmSampler, *, config: dict | None = None
) -> list[TreeInstance]:
    """Measured tree instances (watershed crowns, trunk circle fit); weak ones are 'rejected'."""
    cfg = _vegetation_config(config)
    min_height = float(cfg.get("tree_min_height_m", 2.0))
    min_points = int(cfg.get("tree_min_points", 200))
    chm_raw, grid = canopy_height_model(ps, dtm_sampler)
    if grid is None:
        return []
    chm, mask = _fill_chm(chm_raw)
    detect_min = min(_CROWN_DETECT_MIN_HEIGHT_M, min_height)
    crowns, labels = detect_crowns(chm, mask, grid, min_height=detect_min, return_labels=True)
    if not crowns:
        return []

    xyz = np.asarray(ps.xyz, dtype=np.float64)
    classes = np.asarray(ps.classes)

    def labels_of(idx: np.ndarray) -> np.ndarray:
        cols = np.floor((xyz[idx, 0] - grid.min_x) / grid.pixel_size).astype(int)
        rows = np.floor((grid.max_y - xyz[idx, 1]) / grid.pixel_size).astype(int)
        inside = (rows >= 0) & (rows < grid.height) & (cols >= 0) & (cols < grid.width)
        out = np.zeros(len(idx), dtype=np.int32)
        out[inside] = labels[rows[inside], cols[inside]]
        return out

    tree_idx = np.flatnonzero(classes == TREE)
    trunk_idx = np.flatnonzero(classes == TREE_TRUNK)
    tree_lab = labels_of(tree_idx)
    trunk_lab = labels_of(trunk_idx)

    trees: list[TreeInstance] = []
    used: set[int] = set()
    for crown in sorted(crowns, key=lambda c: (c.trunk[0], c.trunk[1])):
        prow, pcol = grid.local_to_pixel(*crown.trunk)
        label = int(labels[int(round(prow)), int(round(pcol))])
        if label == 0 or label in used:
            continue
        used.add(label)
        own_tree = tree_idx[tree_lab == label]
        own_trunk = trunk_idx[trunk_lab == label]
        notes: dict = {}
        if len(own_trunk) >= _MIN_TRUNK_POINTS:
            trunk_xy = (float(xyz[own_trunk, 0].mean()), float(xyz[own_trunk, 1].mean()))
            notes["trunk_xy"] = "trunk_points"
        else:
            trunk_xy = (float(crown.trunk[0]), float(crown.trunk[1]))
            notes["trunk_xy"] = "crown_peak"
        base = _sample(dtm_sampler, np.array([trunk_xy]))[0]
        indices = np.concatenate([own_tree, own_trunk]).astype(np.int64)
        views = 0
        if ps.view_count is not None and len(indices):
            views = int(np.median(np.asarray(ps.view_count)[indices]))
        evidence = Evidence(views=views, points=int(len(indices)), notes=notes)
        crown_d = 2.0 * math.sqrt(float(np.count_nonzero(labels == label)) * grid.pixel_size ** 2 / math.pi)
        if not np.isfinite(base):
            trees.append(TreeInstance(len(trees), trunk_xy, float("nan"), None, crown_d, None, indices,
                                      evidence, "rejected", "no terrain height at trunk position"))
            continue
        height = float(np.percentile(xyz[own_tree, 2], 99) - base) if len(own_tree) else None
        trunk_d, rms = (None, None)
        if len(own_trunk) >= _MIN_TRUNK_POINTS:
            trunk_d, rms = _trunk_diameter(xyz[own_trunk], trunk_xy, float(base))
        evidence.rmse = rms
        status, reason = "ok", ""
        if height is None or height < min_height:
            status, reason = "rejected", f"height below minimum {min_height:.1f} m"
        elif len(indices) < min_points:
            status, reason = "rejected", f"fewer than {min_points} points"
        trees.append(TreeInstance(len(trees), trunk_xy, float(base), height, crown_d, trunk_d, indices,
                                  evidence, status, reason))
    return trees


def _connected_clusters(xy: np.ndarray, eps: float) -> np.ndarray:
    """Connected-component labels of points closer than eps (DBSCAN with minPts=1)."""
    tree = cKDTree(xy)
    pairs = tree.query_pairs(eps, output_type="ndarray")
    n = len(xy)
    if len(pairs) == 0:
        return np.arange(n)
    graph = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(n, n))
    return connected_components(graph, directed=False)[1]


def detect_shrubs(
    ps: PointSet, dtm_sampler: DtmSampler, *, min_area: float = 0.5, max_height: float = 3.0
) -> list[ShrubInstance]:
    """Shrub clusters (class 94 plus untreed medium vegetation) with concave outline and p95 height."""
    xyz = np.asarray(ps.xyz, dtype=np.float64)
    classes = np.asarray(ps.classes)
    ground = np.full(len(xyz), np.nan)
    cand = np.flatnonzero((classes == SHRUB) | (classes == MEDIUM_VEGETATION))
    if len(cand) == 0:
        return []
    ground[cand] = _sample(dtm_sampler, xyz[cand, :2])
    rel = xyz[:, 2] - ground
    ok = np.isfinite(rel[cand])
    is_medium = classes[cand] == MEDIUM_VEGETATION
    keep = ok & (~is_medium | (rel[cand] < max_height))
    cand = cand[keep]
    if len(cand) == 0:
        return []
    if np.any(classes == TREE):
        chm, grid = canopy_height_model(ps, dtm_sampler)
        if grid is not None:
            _, mask = _fill_chm(chm)
            cols = np.floor((xyz[cand, 0] - grid.min_x) / grid.pixel_size).astype(int)
            rows = np.floor((grid.max_y - xyz[cand, 1]) / grid.pixel_size).astype(int)
            inside = (rows >= 0) & (rows < grid.height) & (cols >= 0) & (cols < grid.width)
            in_tree = np.zeros(len(cand), dtype=bool)
            in_tree[inside] = mask[rows[inside], cols[inside]]
            drop = in_tree & (classes[cand] == MEDIUM_VEGETATION)
            cand = cand[~drop]
    if len(cand) == 0:
        return []
    comp = _connected_clusters(xyz[cand, :2], _SHRUB_EPS_M)
    shrubs: list[ShrubInstance] = []
    for cid in np.unique(comp):
        idx = cand[comp == cid]
        if len(idx) < _SHRUB_MIN_POINTS:
            continue
        hull = concave_hull(MultiPoint(xyz[idx, :2]), ratio=0.3)
        outline = hull if isinstance(hull, Polygon) else Polygon()
        height = float(np.percentile(rel[idx], 95))
        views = 0
        if ps.view_count is not None:
            views = int(np.median(np.asarray(ps.view_count)[idx]))
        evidence = Evidence(views=views, points=int(len(idx)))
        status, reason = "ok", ""
        if outline.area < min_area:
            status, reason = "rejected", f"outline area below {min_area:.2f} m2"
        elif height > max_height:
            status, reason = "rejected", f"height above {max_height:.1f} m (not a shrub)"
        shrubs.append(ShrubInstance(len(shrubs), outline, height, evidence, status, reason))
    return shrubs


def crown_detail_mesh(points: np.ndarray, alpha: float | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Alpha-shape surface (V, F) of crown points for glTF; vertices are input points, no smoothing.

    ``alpha`` is the maximum tetrahedron circumradius in metres; default is
    four times the median nearest-neighbour spacing.
    """
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    empty = (np.zeros((0, 3)), np.zeros((0, 3), dtype=np.int64))
    if len(pts) < 4:
        return empty
    if alpha is None:
        spacing = cKDTree(pts).query(pts, k=2)[0][:, 1]
        alpha = 4.0 * float(np.median(spacing))
    try:
        tets = Delaunay(pts).simplices
    except Exception:  # degenerate (coplanar) input
        logger.warning("crown_detail_mesh: Delaunay failed")
        return empty
    p0, p1, p2, p3 = (pts[tets[:, i]] for i in range(4))
    a, b, c = p1 - p0, p2 - p0, p3 - p0
    vol6 = np.abs(np.einsum("ij,ij->i", a, np.cross(b, c)))
    num = (
        (a ** 2).sum(1)[:, None] * np.cross(b, c)
        + (b ** 2).sum(1)[:, None] * np.cross(c, a)
        + (c ** 2).sum(1)[:, None] * np.cross(a, b)
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        radius = np.linalg.norm(num, axis=1) / (2.0 * vol6)
    kept = tets[np.isfinite(radius) & (radius <= alpha) & (vol6 > 1e-12)]
    if len(kept) == 0:
        return empty
    faces: dict[tuple[int, int, int], list] = {}
    for tet in kept:
        for opp in range(4):
            tri = [int(tet[j]) for j in range(4) if j != opp]
            key = tuple(sorted(tri))
            faces.setdefault(key, []).append((tri, int(tet[opp])))
    out = []
    for entries in faces.values():
        if len(entries) != 1:
            continue
        tri, opp = entries[0]
        normal = np.cross(pts[tri[1]] - pts[tri[0]], pts[tri[2]] - pts[tri[0]])
        if np.dot(normal, pts[opp] - pts[tri[0]]) > 0:
            tri = [tri[0], tri[2], tri[1]]
        out.append(tri)
    if not out:
        return empty
    f = np.asarray(out, dtype=np.int64)
    used, inverse = np.unique(f.ravel(), return_inverse=True)
    return pts[used], inverse.reshape(f.shape).astype(np.int64)


def vegetation_qa(trees: list[TreeInstance], shrubs: list[ShrubInstance]) -> list[QAEntry]:
    """QA entries for every rejected tree or shrub."""
    entries: list[QAEntry] = []
    for t in trees:
        if t.status == "rejected":
            entries.append(QAEntry("tree", str(t.tree_id), t.reason,
                                   {"height_m": t.height_m, "points": t.evidence.points}))
    for s in shrubs:
        if s.status == "rejected":
            entries.append(QAEntry("shrub", str(s.shrub_id), s.reason,
                                   {"height_m": s.height_m, "area_m2": float(s.outline.area),
                                    "points": s.evidence.points}))
    return entries
