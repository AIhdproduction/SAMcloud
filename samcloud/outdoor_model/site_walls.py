"""Survey-derived site walls and fences."""
from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from shapely.geometry import LineString

from samcloud.bim.core.model import FenceElement, FenceParams, ParamOrigin, SiteWallElement, SiteWallParams, Status
from samcloud.drone import contracts as c

LAYER_WALL = "S-WALL"
LAYER_FENCE = "S-FENC"
DtmSampler = Callable[[np.ndarray], np.ndarray]


@dataclass
class SiteWallFeature:
    wall_id: int
    axis: np.ndarray
    thickness: float
    base_z: float
    top_z: float
    evidence: c.Evidence = field(default_factory=c.Evidence)
    status: c.Status = "ok"
    reason: str = ""


@dataclass
class FenceFeature:
    fence_id: int
    axis: np.ndarray
    height: float
    evidence: c.Evidence = field(default_factory=c.Evidence)
    status: c.Status = "ok"
    reason: str = ""


def _components(xyz: np.ndarray, eps: float = 0.75) -> list[np.ndarray]:
    if len(xyz) == 0: return []
    pairs = cKDTree(xyz[:, :2]).query_pairs(eps, output_type="ndarray")
    graph = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(len(xyz), len(xyz)))
    count, labels = connected_components(graph, directed=False)
    return [np.flatnonzero(labels == i) for i in range(count)]


def _dtm(sample: DtmSampler | None, xy: np.ndarray, fallback: float) -> float:
    if sample is None: return fallback
    values = np.asarray(sample(np.asarray(xy, float).reshape(-1, 2)), float)
    values = values[np.isfinite(values)]
    return float(np.median(values)) if len(values) else fallback


def _axis(points: np.ndarray, *, bins: int = 20) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xy = points[:, :2]; centre = xy.mean(0)
    _, vectors = np.linalg.eigh(np.cov((xy - centre).T))
    major = vectors[:, -1]; minor = np.array([-major[1], major[0]])
    u, v = (xy - centre) @ major, (xy - centre) @ minor
    count = max(2, min(bins, int(max(np.ptp(u) / 0.4, 2))))
    edges = np.linspace(u.min(), u.max(), count + 1)
    rows = []
    for left, right in zip(edges[:-1], edges[1:], strict=True):
        sel = (u >= left) & (u <= right)
        if sel.any(): rows.append(centre + major * np.median(u[sel]) + minor * np.median(v[sel]))
    if len(rows) < 2: rows = [centre + major * u.min(), centre + major * u.max()]
    return np.asarray(rows), major, v


def site_wall_features(ps: c.PointSet, dtm_sampler: DtmSampler | None) -> list[SiteWallFeature]:
    """Cluster class-72 points and fit a PCA centre axis and measured wall dimensions."""
    xyz = np.asarray(ps.xyz, float)[np.asarray(ps.classes) == c.SITE_WALL]
    output: list[SiteWallFeature] = []
    for ids in _components(xyz):
        pts = xyz[ids]
        if len(pts) < 4: continue
        axis, _major, cross = _axis(pts)
        thickness = max(0.1, float(np.percentile(cross, 95) - np.percentile(cross, 5)))
        base = _dtm(dtm_sampler, axis, float(np.percentile(pts[:, 2], 5)))
        top = float(np.percentile(pts[:, 2], 95))
        status, reason = ("ok", "") if top > base + 0.05 else ("rejected", "top is not above terrain")
        output.append(SiteWallFeature(len(output), axis, thickness, base, top, c.Evidence(points=len(pts)), status, reason))
    return output


def fence_features(ps: c.PointSet, dtm_sampler: DtmSampler | None) -> list[FenceFeature]:
    """Cluster class-77 points into 3D fence axes and terrain-relative heights."""
    xyz = np.asarray(ps.xyz, float)[np.asarray(ps.classes) == c.FENCE]
    output: list[FenceFeature] = []
    for ids in _components(xyz):
        pts = xyz[ids]
        if len(pts) < 2: continue
        axis_xy, major, _cross = _axis(pts)
        order = np.argsort((axis_xy - axis_xy.mean(0)) @ major)
        axis_xy = axis_xy[order]
        _, nearest = cKDTree(pts[:, :2]).query(axis_xy)
        axis = np.column_stack([axis_xy, pts[nearest, 2]])
        base = _dtm(dtm_sampler, axis_xy, float(np.percentile(pts[:, 2], 5)))
        height = float(np.percentile(pts[:, 2], 95) - base)
        status, reason = ("ok", "") if height > 0.05 else ("rejected", "height is not above terrain")
        output.append(FenceFeature(len(output), axis, height, c.Evidence(points=len(pts)), status, reason))
    return output


def _state(raw: str) -> Status:
    return {"ok": Status.OK, "warning": Status.WARNING}.get(raw, Status.FAIL)


def build_site_elements(
    walls: Sequence[SiteWallFeature], fences: Sequence[FenceFeature], *, config: dict | None = None,
) -> tuple[list[object], list[c.QAEntry]]:
    elements: list[object] = []; qa: list[c.QAEntry] = []
    for wall in walls:
        ident = f"site-wall-{wall.wall_id}"
        if wall.status not in ("ok", "warning"):
            qa.append(c.QAEntry("site_wall", ident, wall.reason or "rejected")); continue
        elements.append(SiteWallElement(
            id=ident, tag=str(wall.wall_id), name="Mauer", params=SiteWallParams(axis=[tuple(map(float, p)) for p in wall.axis],
            thickness=wall.thickness, base_z=wall.base_z, top_z=wall.top_z),
            param_origin={"axis": ParamOrigin.FITTED, "thickness": ParamOrigin.FITTED, "base_z": ParamOrigin.MEASURED, "top_z": ParamOrigin.FITTED},
            evidence={"points": wall.evidence.points, "thickness_m": wall.thickness}, status=_state(wall.status)))
    for fence in fences:
        ident = f"fence-{fence.fence_id}"
        if fence.status not in ("ok", "warning"):
            qa.append(c.QAEntry("fence", ident, fence.reason or "rejected")); continue
        elements.append(FenceElement(
            id=ident, tag=str(fence.fence_id), name="Zaun", params=FenceParams(axis=[tuple(map(float, p)) for p in fence.axis], height=fence.height),
            param_origin={"axis": ParamOrigin.FITTED, "height": ParamOrigin.FITTED}, evidence={"points": fence.evidence.points}, status=_state(fence.status)))
    return elements, qa


def to_dxf_entities(elements: Sequence[object], frame: c.LocalFrame) -> dict[str, list[dict]]:
    output: dict[str, list[dict]] = {}
    for el in elements:
        if isinstance(el, SiteWallElement):
            axis = np.asarray(el.params.axis, float)
            line = LineString(axis)
            poly = line.buffer(float(el.params.thickness) / 2.0, cap_style=2, join_style=2)
            ring = np.asarray(poly.exterior.coords, float)[:-1]
            z = np.full((len(ring), 1), float(el.params.base_z))
            output.setdefault(LAYER_WALL, []).append({"type": "lwpolyline", "points": frame.to_lv95(np.column_stack([ring, z])).tolist(), "closed": True, "elevation": float(frame.origin_lv95[2] + el.params.base_z)})
        elif isinstance(el, FenceElement):
            output.setdefault(LAYER_FENCE, []).append({"type": "polyline3d", "points": frame.to_lv95(np.asarray(el.params.axis, float)).tolist(), "closed": False})
    return output


def category_counts(elements: Sequence[object]) -> dict[str, int]:
    return {"site_wall": sum(isinstance(x, SiteWallElement) for x in elements), "fence": sum(isinstance(x, FenceElement) for x in elements)}
