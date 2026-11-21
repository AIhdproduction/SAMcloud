"""Flush pavement edges: the terrain next to a pavement is cut flush with it, without a lip.

Along every pavement boundary that carries no curb step the shared breakline gets the pavement surface height at the
edge (robust: the DTM of the pavement side, median filtered along the line), so pavement and terrain share exactly
that height there.  The terrain vertices inside a feather band (``terrain.flush.blend_m``, default 1 m) outside the
edge are moved towards the edge height with a smooth (cosine) weight, so a higher or lower terrain meets the pavement
without a step.  Real vertical objects (curbs, walls) are handled by :mod:`samcloud.terrain.steps` and left alone.
"""
from __future__ import annotations

import logging
import math
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field

import numpy as np
import shapely
from scipy import ndimage
from scipy.spatial import cKDTree
from shapely.geometry import LineString
from shapely.ops import unary_union

from samcloud.terrain.steps import _normals, _region_lines, _sample_positions

LOGGER = logging.getLogger(__name__)
PAVEMENT_NAMES = ("road", "sidewalk", "path")
DEFAULTS: dict = {"enabled": True, "blend_m": 1.0, "probe_m": 0.15, "inner_probes_m": (0.2, 0.35, 0.5), "sample_m": 0.25, "step_clear_m": 0.05, "qa_tol_m": 0.01, "max_spread_m": 0.10}


@dataclass
class FlushPlan:
    """Breaklines with finite z and the dense edge samples used for the terrain blend."""

    lines: list[np.ndarray] = field(default_factory=list)
    edge_xy: np.ndarray = field(default_factory=lambda: np.empty((0, 2)))
    edge_z: np.ndarray = field(default_factory=lambda: np.empty(0))
    step_xy: np.ndarray = field(default_factory=lambda: np.empty((0, 2)))
    blend_m: float = 1.0
    report: dict = field(default_factory=dict)


def flush_config(config: Mapping | None) -> dict:
    """``terrain.flush`` of ``config`` merged over the defaults."""
    node = ((config or {}).get("terrain", {}) or {}).get("flush", {}) or {}
    return {**DEFAULTS, **{k: v for k, v in dict(node).items() if v is not None}}


def plan_flush(
    regions: Mapping[str, Iterable], sampler: Callable, step_runs: Iterable = (), config: Mapping | None = None,
) -> FlushPlan:
    """Edge breaklines (z = pavement surface at the edge) for all pavement boundaries without a step."""
    cfg = flush_config(config)
    plan = FlushPlan(blend_m=float(cfg["blend_m"]))
    if not cfg["enabled"]:
        return plan
    polygons = [part for name in PAVEMENT_NAMES for value in regions.get(name, []) or []
                for part in _parts(value)]
    if not polygons:
        return plan
    pavement = unary_union(polygons)
    shapely.prepare(pavement)
    runs = [r for r in step_runs if len(r.xy) >= 2]
    step_geom = unary_union([LineString(r.xy) for r in runs]) if runs else None
    if runs:
        plan.step_xy = np.vstack([r.xy for r in runs])
    probe, clear = float(cfg["probe_m"]), float(cfg["step_clear_m"])
    edge_xy: list[np.ndarray] = []
    edge_z: list[np.ndarray] = []
    total = 0.0
    for line in _region_lines({name: regions.get(name, []) for name in PAVEMENT_NAMES}):
        coords = np.asarray(line.coords)
        closed = len(coords) > 3 and bool(np.allclose(coords[0, :2], coords[-1, :2]))
        positions = _sample_positions(line, float(cfg["sample_m"]))
        if closed:
            positions = positions[:-1]
        if len(positions) < 2:
            continue
        pts = np.asarray([line.interpolate(float(d)).coords[0][:2] for d in positions], dtype=float)
        normals = _normals(pts, closed)
        left, right = pts + probe * normals, pts - probe * normals
        in_left = np.asarray(shapely.contains_xy(pavement, left[:, 0], left[:, 1]), dtype=bool)
        in_right = np.asarray(shapely.contains_xy(pavement, right[:, 0], right[:, 1]), dtype=bool)
        z_left = _inner_height(pavement, sampler, pts, normals, probe, cfg["inner_probes_m"], 1.0, float(cfg["max_spread_m"]))
        z_right = _inner_height(pavement, sampler, pts, normals, probe, cfg["inner_probes_m"], -1.0, float(cfg["max_spread_m"]))
        z = np.where(in_left & in_right, 0.5 * (z_left + z_right), np.where(in_left, z_left, z_right))
        terrain_side = in_left ^ in_right
        usable = (in_left | in_right) & np.isfinite(z)
        finite = np.isfinite(z)
        if not finite.any():
            continue
        z = np.interp(np.arange(len(z)), np.flatnonzero(finite), z[finite])   # gaps are not usable but must not pollute the filter
        z = ndimage.median_filter(z, size=5, mode="wrap" if closed else "nearest")
        if step_geom is not None:
            near = np.asarray(shapely.dwithin(shapely.points(pts[:, 0], pts[:, 1]), step_geom, clear), dtype=bool)
            usable &= ~near
        # contiguous stretches of usable samples become breaklines
        start = None
        for k in range(len(pts) + 1):
            ok = k < len(pts) and usable[k]
            if ok and start is None:
                start = k
            if not ok and start is not None:
                if k - start >= 2:
                    piece = np.column_stack((pts[start:k], z[start:k]))
                    plan.lines.append(piece)
                    total += float(np.linalg.norm(np.diff(piece[:, :2], axis=0), axis=1).sum())
                start = None
        keep = usable & terrain_side
        edge_xy.append(pts[keep])
        edge_z.append(z[keep])
    if edge_xy:
        plan.edge_xy, plan.edge_z = np.vstack(edge_xy), np.concatenate(edge_z)
    plan.report = {"lines": len(plan.lines), "length_m": round(total, 2), "terrain_edge_samples": int(len(plan.edge_z))}
    return plan


def _inner_height(pavement, sampler: Callable, pts: np.ndarray, normals: np.ndarray, probe: float, distances, side: float,
                  max_spread: float = np.inf) -> np.ndarray:
    """Robust pavement surface height next to the edge on one side: median of the DTM at several distances inside the
    pavement (a probe that leaves the pavement is ignored); the plain ``probe`` height when no probe stays inside."""
    stack = []
    for distance in distances:
        q = pts + side * float(distance) * normals
        z = np.asarray(sampler(q), dtype=float).reshape(-1)
        inside = np.asarray(shapely.contains_xy(pavement, q[:, 0], q[:, 1]), dtype=bool)
        stack.append(np.where(inside, z, np.nan))
    values = np.vstack(stack)
    with np.errstate(all="ignore"):
        robust = np.nanmedian(values, axis=0)
        spread = np.nanmax(values, axis=0) - np.nanmin(values, axis=0)
    plain = np.asarray(sampler(pts + side * probe * normals), dtype=float).reshape(-1)
    result = np.where(np.isfinite(robust), robust, plain)
    return np.where(np.isfinite(spread) & (spread > max_spread), np.nan, result)   # no clean pavement surface (wall, slope): unusable


def _parts(value) -> list:
    if hasattr(value, "polygon"):
        value = value.polygon
    if value is None or value.is_empty:
        return []
    if value.geom_type == "Polygon":
        return [value]
    return [p for g in getattr(value, "geoms", ()) for p in _parts(g)]


def blend_terrain(tin, plan: FlushPlan) -> dict:
    """Move the terrain-only vertices within ``blend_m`` of a flush edge towards the edge height (cosine weight).

    Vertices of pavement or step triangles are never moved. Vertices closer to a step than to a flush edge keep their
    height (the step is a real vertical object). Operates in place; returns a small report.
    """
    report = {"moved_vertices": 0, "max_change_m": 0.0}
    if not len(plan.edge_z) or plan.blend_m <= 0 or not len(tin.triangles):
        return report
    vertices = np.asarray(tin.vertices, dtype=float).copy()
    triangles = np.asarray(tin.triangles, dtype=np.int64)
    labels = np.asarray(tin.labels)
    solid = np.zeros(len(vertices), dtype=bool)
    solid[np.unique(triangles[labels != 0])] = True
    distance, index = cKDTree(plan.edge_xy).query(vertices[:, :2], distance_upper_bound=plan.blend_m)
    candidate = np.isfinite(distance) & ~solid
    if len(plan.step_xy):
        d_step, _ = cKDTree(plan.step_xy).query(vertices[:, :2])
        candidate &= d_step >= distance
    weight = np.zeros(len(vertices))
    weight[candidate] = 0.5 * (1.0 + np.cos(math.pi * distance[candidate] / plan.blend_m))
    delta = np.zeros(len(vertices))
    delta[candidate] = weight[candidate] * (plan.edge_z[index[candidate]] - vertices[candidate, 2])
    vertices[:, 2] += delta
    tin.vertices = vertices
    report.update(moved_vertices=int((np.abs(delta) > 1e-6).sum()), max_change_m=round(float(np.abs(delta).max()), 4))
    return report


__all__ = ["FlushPlan", "blend_terrain", "flush_config", "plan_flush"]
