"""Depth evidence as an optional input of the indoor plan (walls, furniture, kitchen).

Owner's rule: dimensions, edges and faces come ONLY from COLMAP-supported voxels (``source & 1``,
``n_views >= 2``); Depth-Anything-only voxels are occupancy evidence ("something is here" / "this is free").
Everything here is optional: :func:`open_plan_depth` returns ``None`` when the switch
``depth_evidence.use_in_plan`` is off or the project has no depth product, and every consumer keeps its previous
behaviour exactly in that case.

All coordinates handed out are plan-local units (``Frame.to_local``), heights are above the level floor, both in
local units (multiply by ``1 / units_per_metre`` for metres), like the rest of the clean-plan code.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import shapely
from shapely.geometry import Polygon

from samcloud.bim.depth_evidence.core import DepthEvidence, _product_dir

DEFAULTS: dict[str, Any] = {
    "use_in_plan": True,
    # Wall faces: class-2 COLMAP voxels without a classified-cloud point nearby, seen from at least this many views.
    "plan_wall_min_views": 2,
    # Furniture / kitchen classes 17 / 18 / 19: gap-fill voxels with this support.
    "plan_furniture_min_views": 2,
    "plan_furniture_min_confidence": 0.5,
    # Two face measurements this close (metres) are the same face.
    "plan_face_tolerance_m": 0.03,
    # Unmeasured cells of the furniture top-height map take the COLMAP furniture top of the product.
    "plan_height_fill": True,
    # ... only within this distance (metres) of cells the cloud measured as an object (no new objects from depth alone).
    "plan_height_fill_reach_m": 0.15,
    # A region is "seen empty" when at least this share of its volume was crossed by rays (min_through views).
    "plan_empty_share": 0.80,
    "plan_min_through": 2,
    # ... and at most this many rays may END in the cell (0: a cell where a ray hit a surface is not empty).
    "plan_free_max_ending": 0,
    # Gap-fill voxels below this height (metres) of a furniture class confirm floor contact.
    "plan_floor_contact_z_m": 0.20,
}

FURNITURE_CLASSES = (17, 18, 19)
WALL_CLASS = 2

_CACHE: dict[tuple[Any, ...], "PlanDepth"] = {}


def plan_settings(overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    """``depth_evidence`` section of ``bim_export.yaml`` (plan keys) merged over the defaults."""
    from samcloud.bim.plan_model import load_bim_export_config

    configured = load_bim_export_config().get("depth_evidence") or {}
    merged = {**DEFAULTS, **{key: value for key, value in configured.items() if key in DEFAULTS}, **(overrides or {})}
    forced = os.environ.get("SAMCLOUD_DEPTH_IN_PLAN", "").strip().lower()  # A/B runs: 0 / false = off, 1 / true = on
    if forced in {"0", "false", "no", "off"}:
        merged["use_in_plan"] = False
    elif forced in {"1", "true", "yes", "on"}:
        merged["use_in_plan"] = True
    return merged


def plan_depth_requested(overrides: dict[str, Any] | None = None) -> bool:
    return bool(plan_settings(overrides).get("use_in_plan", False))


def plan_depth_fingerprint(project_dir: Path | str, overrides: dict[str, Any] | None = None) -> str | None:
    """Short digest of the depth settings + product when depth is used in the plan, else ``None`` (no cache-key change)."""
    settings = plan_settings(overrides)
    if not settings.get("use_in_plan"):
        return None
    product = _product_dir(Path(project_dir))
    points, free = product / "depth_points.npz", product / "free_space.npz"
    if not (points.is_file() and free.is_file()):
        return None
    digest = hashlib.sha256()
    for key in sorted(settings):
        digest.update(f"{key}={settings[key]}".encode())
    for path in (points, free):
        stat = path.stat()
        digest.update(f"{path.name}:{stat.st_size}:{stat.st_mtime_ns}".encode())
    return digest.hexdigest()[:16]


@dataclass(slots=True)
class PlanDepth:
    """Plan-frame view of one project's depth product with the subsets the plan code needs, computed once."""

    evidence: DepthEvidence
    floor_z: float
    units_per_metre: float
    settings: dict[str, Any]
    _free: np.ndarray | None = None  # (M, 3) plan xyz of seen-empty voxel centres, z above floor
    _voxel_plan: float = 0.05

    # -- subsets -----------------------------------------------------------------------------------------
    def _subset(self, classes: Iterable[int], *, gap_only: bool, min_views: int, min_confidence: float, colmap_only: bool) -> tuple[np.ndarray, np.ndarray]:
        mask = self.evidence.select(tuple(classes), gap_only=gap_only, min_views=min_views, min_confidence=min_confidence, colmap_only=colmap_only)
        index = np.flatnonzero(mask)
        points = self.evidence.plan_points()[index].astype(np.float64)
        points[:, 2] -= self.floor_z
        return points, index

    def wall_points(self, z_low: float, z_high: float) -> np.ndarray:
        """(N, 3) class-2 COLMAP gap-fill voxels (x, y, z above floor) in the wall band ``z_low..z_high`` (local units)."""
        points, _ = self._subset(
            (WALL_CLASS,), gap_only=True, min_views=int(self.settings["plan_wall_min_views"]), min_confidence=0.0, colmap_only=True,
        )
        keep = (points[:, 2] >= z_low) & (points[:, 2] <= z_high)
        return points[keep]

    def furniture_points(self, *, colmap_only: bool, classes: Iterable[int] = FURNITURE_CLASSES, gap_only: bool = True) -> tuple[np.ndarray, np.ndarray]:
        """(points (N, 3) with z above floor, class ids) of furniture-class voxels (gap-fill, >= min views / confidence)."""
        points, index = self._subset(
            classes, gap_only=gap_only, min_views=int(self.settings["plan_furniture_min_views"]),
            min_confidence=float(self.settings["plan_furniture_min_confidence"]), colmap_only=colmap_only,
        )
        return points, self.evidence.class_id[index]

    def top_height_map(self, grid: Any, *, z_low: float, z_high: float, min_voxels: int = 3, percentile: float = 0.95, classes: Iterable[int] = FURNITURE_CLASSES) -> np.ndarray:
        """Per raster cell the robust top (p95) of COLMAP furniture-class gap-fill voxels, local units above floor (NaN = none).

        Cells with fewer than ``min_voxels`` voxels are ignored (single stray voxels do not make a surface).
        """
        points, _ = self.furniture_points(colmap_only=True, classes=classes)
        top = np.full((grid.height, grid.width), np.nan, dtype=np.float32)
        if not len(points):
            return top
        keep = (points[:, 2] >= z_low) & (points[:, 2] <= z_high)
        points = points[keep]
        rows = np.floor((grid.max_y - points[:, 1]) / grid.pixel_size).astype(np.int64)
        cols = np.floor((points[:, 0] - grid.min_x) / grid.pixel_size).astype(np.int64)
        inside = (rows >= 0) & (rows < grid.height) & (cols >= 0) & (cols < grid.width)
        rows, cols, z = rows[inside], cols[inside], points[inside, 2]
        if not len(z):
            return top
        cell = rows * grid.width + cols
        order = np.lexsort((z, cell))
        cell, z = cell[order], z[order]
        starts = np.r_[0, np.flatnonzero(np.diff(cell)) + 1]
        counts = np.diff(np.r_[starts, len(cell)])
        pick = starts + np.floor(percentile * (counts - 1)).astype(np.int64)
        valid = counts >= int(min_voxels)
        flat = top.reshape(-1)
        flat[cell[starts[valid]]] = z[pick[valid]].astype(np.float32)
        return top

    # -- seen-empty space --------------------------------------------------------------------------------
    def _free_points(self) -> np.ndarray:
        if self._free is None:
            evidence = self.evidence
            centres = evidence.free_origin + (evidence.free_indices.astype(np.float32) + 0.5) * evidence.free_voxel
            # Seen empty = rays crossed the cell and NONE ended in it (measured on v12: 'through > ending' leaves 88 % of the sofa and 40 % of the
            # wall points inside "free" cells, 'ending == 0' leaves <= 8 %).
            keep = (evidence.free_through >= int(self.settings["plan_min_through"])) & (evidence.free_ending <= int(self.settings["plan_free_max_ending"]))
            local = np.asarray(evidence._to_plan(centres[keep]), dtype=np.float64)
            local[:, 2] -= self.floor_z
            self._free = local
            scale = 1.0 if evidence.frame is None else float(getattr(evidence.frame, "scale_factor", 1.0))
            self._voxel_plan = float(evidence.free_voxel) * scale
        return self._free

    def free_share(self, polygon: Polygon, z_low: float, z_high: float) -> float:
        """Share (0..1) of the volume ``polygon x [z_low, z_high]`` (local units) that rays crossed (seen empty)."""
        if polygon is None or polygon.is_empty or z_high <= z_low:
            return 0.0
        free = self._free_points()
        minx, miny, maxx, maxy = polygon.bounds
        box = (free[:, 0] >= minx) & (free[:, 0] <= maxx) & (free[:, 1] >= miny) & (free[:, 1] <= maxy) & (free[:, 2] >= z_low) & (free[:, 2] <= z_high)
        index = np.flatnonzero(box)
        if not len(index):
            return 0.0
        inside = shapely.contains_xy(polygon, free[index, 0], free[index, 1])
        voxel = self._voxel_plan
        expected = polygon.area * (z_high - z_low) / (voxel ** 3)
        return float(min(1.0, int(inside.sum()) / max(expected, 1.0)))

    def is_seen_empty(self, polygon: Polygon, z_low: float, z_high: float) -> bool:
        return self.free_share(polygon, z_low, z_high) >= float(self.settings["plan_empty_share"])

    @staticmethod
    def points_inside(polygon: Polygon, points: np.ndarray, z_low: float | None = None, z_high: float | None = None) -> np.ndarray:
        """Boolean mask of ``points`` (N, 3) inside ``polygon`` (optionally within a z range)."""
        if not len(points):
            return np.zeros(0, bool)
        mask = shapely.contains_xy(polygon, points[:, 0], points[:, 1])
        if z_low is not None:
            mask &= points[:, 2] >= z_low
        if z_high is not None:
            mask &= points[:, 2] <= z_high
        return mask


def open_plan_depth(project_dir: Path | str, frame: Any, floor_z: float, units_per_metre: float, overrides: dict[str, Any] | None = None) -> PlanDepth | None:
    """The project's depth evidence in plan coordinates, or ``None`` (switch off / no product).

    One instance per project and frame is kept in memory, so the two clean-plan passes of a chain open the product once.
    Never raises: a damaged product is a missing product.
    """
    settings = plan_settings(overrides)
    if not settings.get("use_in_plan"):
        return None
    project = Path(project_dir)
    product = _product_dir(project)
    points, free = product / "depth_points.npz", product / "free_space.npz"
    if not (points.is_file() and free.is_file()):
        return None
    key = (
        str(project.resolve()), points.stat().st_mtime_ns, free.stat().st_mtime_ns, tuple(np.asarray(frame.origin, float).round(6)),
        round(float(frame.scale_factor), 9), round(float(floor_z), 6), tuple(sorted(settings.items())),
    )
    cached = _CACHE.get(key)
    if cached is not None:
        return cached
    try:
        evidence = DepthEvidence(product, frame)
    except Exception:  # noqa: BLE001 - optional input
        return None
    _CACHE.clear()  # one project at a time keeps the memory bounded
    plan = PlanDepth(evidence=evidence, floor_z=float(floor_z), units_per_metre=float(units_per_metre), settings=settings)
    _CACHE[key] = plan
    return plan
