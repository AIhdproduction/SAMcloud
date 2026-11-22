"""Scan points of one furniture item: footprint selection, ownership between overlapping items, vertical runs, cleaning."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import shapely
from shapely.geometry import Polygon

from samcloud.bim.furniture_recon.families import family_of
from samcloud.bim.furniture_recon.settings import ReconSettings


@dataclass(slots=True)
class SceneCloud:
    """Furniture-class points of the classified cloud in plan-local coordinates (absolute local z)."""

    xyz: np.ndarray
    fine: np.ndarray  # fine term id per point (0 = none)
    instance: np.ndarray  # instance id per point (0 = none)
    fine_names: list[str]

    def fine_families(self) -> np.ndarray:
        """Reconstruction family index per fine-term id (lookup table, index 0 = none)."""
        return np.asarray(["none"] + [family_of(name, name, None) for name in self.fine_names], dtype=object)


def load_scene_cloud(ply_path: Path | str, frame: Any, classes: tuple[int, ...]) -> SceneCloud | None:
    """Furniture-class points of a classified PLY (``scalar_class``; ``fine_term`` / ``instance_id`` when present)."""
    path = Path(ply_path)
    if not path.is_file() or path.suffix.lower() != ".ply":
        return None
    from samcloud.engine.ply_io import read_vertex_array

    vertices = read_vertex_array(path)
    names = vertices.dtype.names or ()
    if "scalar_class" not in names:
        return None
    keep = np.isin(vertices["scalar_class"], np.asarray(classes, dtype=np.uint8))
    if not keep.any():
        return None
    xyz = np.column_stack((vertices["x"][keep], vertices["y"][keep], vertices["z"][keep])).astype(np.float64)
    fine = np.asarray(vertices["fine_term"][keep], dtype=np.int32) if "fine_term" in names else np.zeros(int(keep.sum()), np.int32)
    instance = np.asarray(vertices["instance_id"][keep], dtype=np.int64) if "instance_id" in names else np.zeros(int(keep.sum()), np.int64)
    fine_names: list[str] = []
    labels_file = path.with_name("fine_labels.json")
    if labels_file.is_file():
        try:
            fine_names = [str(name).strip().lower() for name in json.loads(labels_file.read_text(encoding="utf-8")).get("labels", [])]
        except (OSError, ValueError):
            fine_names = []
    if fine_names:
        fine[(fine < 0) | (fine > len(fine_names))] = 0
    else:
        fine[:] = 0
    local = np.asarray(frame.to_local(xyz), dtype=np.float64)
    return SceneCloud(local, fine, instance, fine_names)


@dataclass(slots=True)
class ItemDescriptor:
    """What the point selection needs to know about one plan item."""

    key: str
    polygon: Polygon  # local units
    family: str
    instance_id: int | None


def select_indices(cloud: SceneCloud, polygon: Polygon, margin: float) -> np.ndarray:
    grown = polygon.buffer(margin)
    x0, y0, x1, y1 = grown.bounds
    pts = cloud.xyz
    near = np.flatnonzero((pts[:, 0] >= x0) & (pts[:, 0] <= x1) & (pts[:, 1] >= y0) & (pts[:, 1] <= y1))
    if not len(near):
        return near
    inside = shapely.contains_xy(grown, pts[near, 0], pts[near, 1])
    return near[inside]


def owned_indices(cloud: SceneCloud, item: ItemDescriptor, others: list[ItemDescriptor], margin: float, fine_family: np.ndarray) -> np.ndarray:
    """Points inside the item's footprint that do not clearly belong to an overlapping other item.

    A point is given to the other item when it lies inside that item's footprint and its fine term names the other item's
    family (a chair under a table) or it carries the other item's instance id.
    """
    indices = select_indices(cloud, item.polygon, margin)
    if not len(indices):
        return indices
    grown = item.polygon.buffer(margin)
    drop = np.zeros(len(indices), dtype=bool)
    families = fine_family[cloud.fine[indices]]
    # a point carrying another plan item's instance id is that item's, wherever it lies (a neighbour's face in the margin)
    foreign = sorted({int(o.instance_id) for o in others if o.key != item.key and o.instance_id is not None and o.instance_id != item.instance_id})
    if foreign:
        drop |= np.isin(cloud.instance[indices], np.asarray(foreign))
    for other in others:
        if other.key == item.key or not other.polygon.buffer(margin).intersects(grown):
            continue
        inside = shapely.contains_xy(other.polygon.buffer(margin), cloud.xyz[indices, 0], cloud.xyz[indices, 1])
        if not inside.any():
            continue
        if other.family != item.family:
            drop |= inside & (families == other.family)
        if other.instance_id is not None and other.instance_id != item.instance_id:
            drop |= inside & (cloud.instance[indices] == int(other.instance_id))
    # Points whose fine term names another object family (bed edge in a nightstand footprint, table top over a chair) are not
    # the item's own when the item's own family is the stronger label inside the footprint (fine terms are noisy, so a
    # footprint whose own points mostly carry another label keeps them).
    own = int(np.count_nonzero(families == item.family))
    if own:
        for name in np.unique(families):
            if name in ("none", "auto", item.family):
                continue
            foreign = families == name
            if int(np.count_nonzero(foreign)) < own:
                drop |= foreign
    return indices[~drop]


def vertical_runs(z: np.ndarray, gap: float, bin_m: float = 0.02) -> list[tuple[float, float, int]]:
    """Height runs (z_low, z_high, count) separated by bands of at least ``gap`` with (almost) no points."""
    if not len(z):
        return []
    low, high = float(z.min()), float(z.max())
    edges = np.arange(low, high + bin_m, bin_m)
    if len(edges) < 2:
        return [(low, high, int(len(z)))]
    counts, edges = np.histogram(z, bins=edges)
    empty_limit = max(1, int(0.001 * len(z)))
    occupied = counts > empty_limit
    runs: list[tuple[int, int]] = []
    start = None
    empty_bins = 0
    need = max(int(round(gap / bin_m)), 1)
    last_occupied = None
    for index, flag in enumerate(occupied):
        if flag:
            if start is None:
                start = index
            elif empty_bins >= need:
                runs.append((start, last_occupied))
                start = index
            empty_bins = 0
            last_occupied = index
        else:
            empty_bins += 1
    if start is not None and last_occupied is not None:
        runs.append((start, last_occupied))
    result = []
    for a, b in runs:
        z0, z1 = float(edges[a]), float(edges[b + 1])
        result.append((z0, z1, int(((z >= z0) & (z <= z1)).sum())))
    return result


def choose_runs(z: np.ndarray, gap: float, tall_gap: float, standing: bool, tall: bool, anchor_floor: bool = False) -> tuple[float, float] | None:
    """Height band of the object: the run with most points, the runs below it (standing objects: supports, plinth) and
    the runs above it that follow within the allowed gap."""
    runs = vertical_runs(z, gap)
    if not runs:
        return None
    main = max(range(len(runs)), key=lambda i: runs[i][2])
    if anchor_floor and runs[0][0] <= 0.15:
        main = 0  # a low piece is the run standing on the floor; a wall TV above a gap is another object
    low, high = runs[main][0], runs[main][1]
    allowed = tall_gap if tall else gap
    if standing:
        low = runs[0][0]
    else:
        for index in range(main - 1, -1, -1):
            if low - runs[index][1] < allowed - 1e-9:
                low = runs[index][0]
            else:
                break
    for index in range(main + 1, len(runs)):
        if runs[index][0] - high < allowed - 1e-9:
            high = runs[index][1]
        else:
            break
    return low, high


def remove_outliers(points: np.ndarray, k: int, std_ratio: float) -> np.ndarray:
    """Statistical outlier removal (mean distance to the k nearest neighbours)."""
    if len(points) <= k + 2:
        return points
    from scipy.spatial import cKDTree

    distances, _ = cKDTree(points).query(points, k=k + 1)
    mean = distances[:, 1:].mean(axis=1)
    limit = float(mean.mean() + std_ratio * mean.std())
    return points[mean <= limit]


def voxel_downsample(points: np.ndarray, voxel: float, max_points: int) -> np.ndarray:
    if not len(points):
        return points
    keys = np.floor(points / voxel).astype(np.int64)
    _, first = np.unique(keys, axis=0, return_index=True)
    reduced = points[np.sort(first)]
    if len(reduced) > max_points:
        rng = np.random.default_rng(0)
        reduced = reduced[np.sort(rng.choice(len(reduced), max_points, replace=False))]
    return reduced


def clean_item_points(
    item_xyz: np.ndarray, window: tuple[float, float], settings: ReconSettings, *, standing: bool, tall: bool, anchor_floor: bool = False,
) -> np.ndarray:
    """Metric item points (item frame, z above floor) -> the cleaned points of the object itself."""
    pts = np.asarray(item_xyz, dtype=float)
    if not len(pts):
        return pts
    pts = pts[(pts[:, 2] >= window[0] - 0.02) & (pts[:, 2] <= window[1])]
    if len(pts) < 8:
        return pts
    pts = voxel_downsample(pts, settings.voxel_m, settings.max_points)
    pts = remove_outliers(pts, settings.outlier_k, settings.outlier_std_ratio)
    band = choose_runs(pts[:, 2], settings.run_gap_m, settings.tall_run_gap_m, standing, tall, anchor_floor)
    if band is not None:
        pts = pts[(pts[:, 2] >= band[0] - 1e-9) & (pts[:, 2] <= band[1] + 1e-9)]
    return pts


def xy_coverage(points: np.ndarray, polygon_item: Polygon, cell: float = 0.05) -> float:
    """Share of the footprint cells (item frame) that hold at least one point seen from above."""
    if not len(points) or polygon_item.is_empty:
        return 0.0
    x0, y0, x1, y1 = polygon_item.bounds
    xs = np.arange(x0 + cell / 2, x1, cell)
    ys = np.arange(y0 + cell / 2, y1, cell)
    if not len(xs) or not len(ys):
        return 1.0
    gx, gy = np.meshgrid(xs, ys)
    flat = np.column_stack((gx.ravel(), gy.ravel()))
    inside = shapely.contains_xy(polygon_item, flat[:, 0], flat[:, 1])
    if not inside.any():
        return 1.0
    cells = {(int(a), int(b)) for a, b in np.floor((points[:, :2] - (x0, y0)) / cell).astype(int)}
    keys = np.floor((flat[inside] - (x0, y0)) / cell).astype(int)
    hit = sum((int(a), int(b)) in cells for a, b in keys)
    return float(hit / len(keys))


def grown_indices(
    cloud: SceneCloud, item: ItemDescriptor, others: list[ItemDescriptor], core: np.ndarray, grow: float, link: float, fine_family: np.ndarray,
    own_terms: set[str] | None = None, extent_margin: float | None = None,
) -> np.ndarray:
    """Points of the item's own fine term just outside its plan footprint (up to ``grow``) that are spatially connected
    (chains of ``link`` steps) to the item's own-labelled core points: a table top or wardrobe the plan outline cut short.

    Never grown in: points inside another item's outline, points carrying another item's instance id, points of
    another fine term, and (when the item has an instance) points outside the robust 3D extent of that instance grown
    by ``extent_margin``."""
    if not len(core) or grow <= 0:
        return np.empty(0, dtype=np.int64)
    ring = select_indices(cloud, item.polygon, grow)
    ring = np.setdiff1d(ring, core, assume_unique=False)
    if not len(ring):
        return ring
    ids = [index + 1 for index, name in enumerate(cloud.fine_names) if own_terms and name in own_terms]
    if ids:
        # the item's own fine term (a wardrobe grows over wardrobe points, not over the shelf beside it of the same family)
        own = np.isin(cloud.fine[ring], np.asarray(ids))
    else:
        own = fine_family[cloud.fine[ring]] == item.family
    ring = ring[own]
    foreign_instances = {int(o.instance_id) for o in others if o.key != item.key and o.instance_id is not None and o.instance_id != item.instance_id}
    if foreign_instances and len(ring):
        ring = ring[~np.isin(cloud.instance[ring], np.asarray(sorted(foreign_instances)))]
    for other in others:
        if other.key == item.key or not len(ring):
            continue
        inside = shapely.contains_xy(other.polygon, cloud.xyz[ring, 0], cloud.xyz[ring, 1])
        ring = ring[~inside]
    if item.instance_id is not None and extent_margin is not None and len(ring):
        member = cloud.xyz[cloud.instance == int(item.instance_id)]
        if len(member) >= 30:
            low = np.percentile(member, 2, axis=0) - extent_margin
            high = np.percentile(member, 98, axis=0) + extent_margin
            ring = ring[np.all((cloud.xyz[ring] >= low) & (cloud.xyz[ring] <= high), axis=1)]
    if not len(ring):
        return ring
    from scipy.spatial import cKDTree

    seeds = core
    if ids:
        own_core = core[np.isin(cloud.fine[core], np.asarray(ids))]
        if len(own_core) >= 20:
            seeds = own_core  # continuity through the item's own points, not through a neighbour lying in its margin
    tree = cKDTree(cloud.xyz[ring])
    reached = np.zeros(len(ring), dtype=bool)
    frontier = voxel_downsample(cloud.xyz[seeds], link / 2.0, 20000)
    for _ in range(200):
        hits = tree.query_ball_point(frontier, r=link)
        new = np.unique(np.fromiter((i for group in hits for i in group), dtype=np.int64)) if len(hits) else np.empty(0, np.int64)
        new = new[~reached[new]] if len(new) else new
        if not len(new):
            break
        reached[new] = True
        frontier = voxel_downsample(cloud.xyz[ring[new]], link / 2.0, 20000)
    return ring[reached]
