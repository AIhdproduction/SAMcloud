"""Signed point-to-surface deviation, coverage and status (MASTERPROMPT 7).

Every modeled face has a normal that points to the side it is observed from. The signed distance
of a point is positive in front of the face (on the normal side) and negative behind it. The
distance of a point to an element is the distance to its nearest face. A face is a planar polygon
(openings and holes removed): points whose projection lies outside the polygon are measured to
the nearest polygon point, so edges and holes do not create false zero distances.

Faces come from the geometry module (docs/GEOMETRY.md):

- wall: both side faces, normals pointing out of the wall body (toward the room), clipped to the
  trimmed wall and minus the opening cuts;
- slab (floor): the top face, normal up;
- ceiling: the underside including a tilted plane, normal down;
- opening, window, door: the two faces of the opening box, normals pointing out of the box.

An ``observer`` position overrides the normals: each face is flipped so that its normal points
toward the observer. Furniture (one-sided deviation) is not covered here.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import shapely
from numpy.typing import NDArray
from scipy.spatial import cKDTree
from shapely.geometry import Polygon, box
from shapely.ops import unary_union

from samcloud.bim.core.geometry import SceneGeometry
from samcloud.bim.core.model import Metrics, Scene, Status
from samcloud.bim.core.qa.tolerances import ToleranceColors

Array = NDArray[np.float64]
Vec3 = tuple[float, float, float]


@dataclass(frozen=True)
class Face:
    """Planar face: origin, unit normal (observed side), axes u, v and a polygon in (u, v)."""

    label: str
    origin: Array
    normal: Array
    u: Array
    v: Array
    polygon: Polygon

    def flipped(self) -> Face:
        """The same face seen from the other side (normal reversed)."""
        return Face(self.label, self.origin, -self.normal, self.u, self.v, self.polygon)

    def to_uv(self, points: Array) -> Array:
        rel = points - self.origin
        return np.column_stack((rel @ self.u, rel @ self.v))

    def to_xyz(self, uv: Array) -> Array:
        xyz: Array = self.origin + uv[:, :1] * self.u + uv[:, 1:2] * self.v
        return xyz


@dataclass(frozen=True)
class DeviationResult:
    """Deviation of the points of one element."""

    element_id: str
    signed_distance: Array
    """Signed distance per input point (m); positive in front of the model surface."""
    face_index: NDArray[np.int64]
    """Index of the nearest face per input point."""
    metrics: Metrics
    status: Status
    tolerance: float
    coverage_per_face: tuple[float, ...] = ()
    unsure_flags: tuple[str, ...] = ()


@dataclass(frozen=True)
class DeviationSettings:
    """Parameters of the deviation computation."""

    coverage_radius: float = 0.05
    coverage_min: float = 0.3
    max_grid_samples: int = 2_000_000


def _unit(vec: Sequence[float]) -> Array:
    arr = np.asarray(vec, dtype=np.float64)
    return arr / np.linalg.norm(arr)


def _frame_from_normal(normal: Array) -> tuple[Array, Array]:
    """Orthonormal axes ``u`` (horizontal where possible) and ``v = n x u``."""
    ref = np.array([1.0, 0.0, 0.0]) if abs(normal[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = _unit(ref - normal * float(ref @ normal))
    return u, np.cross(normal, u)


def _wall_faces(geometry: SceneGeometry, element_id: str) -> list[Face]:
    wall = geometry.walls[element_id]
    d = np.array([wall.direction[0], wall.direction[1], 0.0])
    n = np.array([wall.normal[0], wall.normal[1], 0.0])
    z_axis = np.array([0.0, 0.0, 1.0])
    start = np.array([wall.param_start[0], wall.param_start[1], 0.0])
    holes = [box(cut.s_min, cut.z_min, cut.s_max, cut.z_max) for cut in wall.opening_cuts]
    hole_union = unary_union(holes) if holes else None
    footprint = wall.footprint
    sides = (
        ("right", -n, wall.offset_right, footprint[0], footprint[1]),
        ("left", n, wall.offset_left, footprint[3], footprint[2]),
    )
    faces: list[Face] = []
    for label, normal, offset, p_a, p_b in sides:
        s_a = (p_a[0] - wall.param_start[0]) * d[0] + (p_a[1] - wall.param_start[1]) * d[1]
        s_b = (p_b[0] - wall.param_start[0]) * d[0] + (p_b[1] - wall.param_start[1]) * d[1]
        polygon = box(min(s_a, s_b), wall.base_z, max(s_a, s_b), wall.top_z)
        if hole_union is not None:
            polygon = polygon.difference(hole_union)
        faces.append(Face(f"wall_{label}", start + n * offset, normal, d, z_axis, polygon))
    return faces


def _slab_face(geometry: SceneGeometry, element_id: str) -> list[Face]:
    slab = geometry.slabs[element_id]
    first = slab.outline[0]
    origin = np.array([first[0], first[1], slab.top_z])
    u, v = np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])
    ring = [(x - first[0], y - first[1]) for x, y in slab.outline]
    holes = [[(x - first[0], y - first[1]) for x, y in hole] for hole in slab.holes]
    polygon = Polygon(ring, holes)
    return [Face("slab_top", origin, np.array([0.0, 0.0, 1.0]), u, v, polygon)]


def _ceiling_face(geometry: SceneGeometry, element_id: str) -> list[Face]:
    ceiling = geometry.ceilings[element_id]
    if ceiling.plane_normal is not None and ceiling.plane_offset is not None:
        normal = _unit(ceiling.plane_normal)
        if normal[2] > 0:
            normal = -normal
    else:
        normal = np.array([0.0, 0.0, -1.0])
    first = ceiling.outline[0]
    origin = np.array([first[0], first[1], ceiling.bottom_z_at(first[0], first[1])])
    u, v = _frame_from_normal(normal)
    vertices = np.array(
        [[x, y, ceiling.bottom_z_at(x, y)] for x, y in ceiling.outline], dtype=np.float64
    )
    face = Face("ceiling_bottom", origin, normal, u, v, Polygon())
    polygon = Polygon(face.to_uv(vertices))
    return [Face("ceiling_bottom", origin, normal, u, v, polygon)]


def _opening_faces(geometry: SceneGeometry, element_id: str) -> list[Face]:
    opening_id = element_id
    if element_id in geometry.windows:
        opening_id = geometry.windows[element_id].opening_id
    elif element_id in geometry.doors:
        opening_id = geometry.doors[element_id].opening_id
    opening = geometry.openings[opening_id]
    u = np.asarray(opening.u_axis, dtype=np.float64)
    v = np.asarray(opening.v_axis, dtype=np.float64)
    w = np.asarray(opening.w_axis, dtype=np.float64)
    origin = np.asarray(opening.origin, dtype=np.float64)
    polygon = box(0.0, 0.0, opening.width, opening.height)
    return [
        Face("opening_front", origin, -w, u, v, polygon),
        Face("opening_back", origin + w * opening.depth, w, u, v, polygon),
    ]


def element_faces(
    geometry: SceneGeometry, element_id: str, observer: Vec3 | None = None
) -> list[Face]:
    """Modeled faces of an element; see the module docstring for the normals.

    With ``observer`` every face is flipped to point toward that position.
    """
    if element_id in geometry.walls:
        faces = _wall_faces(geometry, element_id)
    elif element_id in geometry.slabs:
        faces = _slab_face(geometry, element_id)
    elif element_id in geometry.ceilings:
        faces = _ceiling_face(geometry, element_id)
    elif (
        element_id in geometry.openings
        or element_id in geometry.windows
        or (element_id in geometry.doors)
    ):
        faces = _opening_faces(geometry, element_id)
    else:
        raise KeyError(f"element {element_id!r} has no deviation geometry")
    if observer is not None:
        position = np.asarray(observer, dtype=np.float64)
        faces = [
            face.flipped() if float((position - face.origin) @ face.normal) < 0 else face
            for face in faces
        ]
    return faces


def signed_distances(points: Array, faces: Sequence[Face]) -> tuple[Array, NDArray[np.int64]]:
    """Signed distance of each point to its nearest face, and the index of that face."""
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    if len(pts) == 0:
        return np.empty(0), np.empty(0, dtype=np.int64)
    abs_dist = np.empty((len(faces), len(pts)))
    side = np.empty((len(faces), len(pts)))
    for index, face in enumerate(faces):
        rel = pts - face.origin
        normal_dist = rel @ face.normal
        uv = np.column_stack((rel @ face.u, rel @ face.v))
        shapely.prepare(face.polygon)
        inside = shapely.contains_xy(face.polygon, uv[:, 0], uv[:, 1]) | shapely.intersects_xy(
            face.polygon, uv[:, 0], uv[:, 1]
        )
        in_plane = np.zeros(len(pts))
        outside = np.flatnonzero(~inside)
        if len(outside):
            in_plane[outside] = shapely.distance(face.polygon, shapely.points(uv[outside]))
        abs_dist[index] = np.hypot(normal_dist, in_plane)
        side[index] = np.where(normal_dist < 0, -1.0, 1.0)
    nearest = np.argmin(abs_dist, axis=0).astype(np.int64)
    columns = np.arange(len(pts))
    return abs_dist[nearest, columns] * side[nearest, columns], nearest


def face_coverage(face: Face, points: Array, radius: float, max_samples: int = 2_000_000) -> float:
    """Fraction of the face surface (grid spacing ``radius / 2``) with a point within ``radius``."""
    if len(points) == 0:
        return 0.0
    spacing = radius / 2.0
    min_u, min_v, max_u, max_v = face.polygon.bounds
    nu = max(int(math.ceil((max_u - min_u) / spacing)), 1)
    nv = max(int(math.ceil((max_v - min_v) / spacing)), 1)
    if nu * nv > max_samples:
        raise ValueError(f"coverage grid of {nu * nv} samples exceeds the limit {max_samples}")
    gu = min_u + (np.arange(nu) + 0.5) * ((max_u - min_u) / nu)
    gv = min_v + (np.arange(nv) + 0.5) * ((max_v - min_v) / nv)
    grid_u, grid_v = np.meshgrid(gu, gv, indexing="ij")
    uv = np.column_stack((grid_u.ravel(), grid_v.ravel()))
    shapely.prepare(face.polygon)
    uv = uv[shapely.contains_xy(face.polygon, uv[:, 0], uv[:, 1])]
    if len(uv) == 0:
        rep = face.polygon.representative_point()
        uv = np.array([[rep.x, rep.y]])
    samples = face.to_xyz(uv)
    shift = samples.mean(axis=0)
    tree = cKDTree(np.asarray(points, dtype=np.float64) - shift)
    nearest, _ = tree.query(samples - shift, k=1, distance_upper_bound=radius * (1 + 1e-12))
    return float(np.mean(np.isfinite(nearest)))


def status_from_metrics(
    metrics: Metrics,
    tolerance: float,
    coverage_min: float,
    unsure_flags: Sequence[str] = (),
) -> Status:
    """Status per MASTERPROMPT 7.

    - ok: p95 at most ``tolerance`` and coverage at least ``coverage_min``;
    - warning: p95 up to 2 times ``tolerance``, or coverage below ``coverage_min``;
    - fail: above that (p95 over 2 times ``tolerance``);
    - unsure: only when ``unsure_flags`` (contradicting evidence from scripts or an unsure AI
      answer) is not empty. Missing metrics (no points) never give ok.
    """
    if unsure_flags:
        return Status.UNSURE
    p95 = metrics.deviation_p95
    coverage = metrics.coverage if metrics.coverage is not None else 0.0
    if p95 is None:
        return Status.WARNING
    if p95 > 2.0 * tolerance:
        return Status.FAIL
    if p95 <= tolerance and coverage >= coverage_min:
        return Status.OK
    return Status.WARNING


def compute_deviation(
    geometry: SceneGeometry,
    element_id: str,
    points: Array,
    tolerance: float,
    settings: DeviationSettings | None = None,
    observer: Vec3 | None = None,
    unsure_flags: Sequence[str] = (),
) -> DeviationResult:
    """Deviation metrics and status of the points assigned to one element.

    ``coverage`` of the element is the highest coverage of its faces (a wall counts as covered
    when at least one side was observed); the coverage per face is returned as well.
    """
    settings = settings or DeviationSettings()
    faces = element_faces(geometry, element_id, observer)
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    signed, nearest = signed_distances(pts, faces)
    coverages = tuple(
        face_coverage(face, pts, settings.coverage_radius, settings.max_grid_samples)
        for face in faces
    )
    if len(pts):
        absolute = np.abs(signed)
        metrics = Metrics(
            deviation_rms=float(np.sqrt(np.mean(signed**2))),
            deviation_p95=float(np.quantile(absolute, 0.95)),
            deviation_max=float(absolute.max()),
            coverage=max(coverages),
            point_count=len(pts),
        )
    else:
        metrics = Metrics(coverage=0.0, point_count=0)
    status = status_from_metrics(metrics, tolerance, settings.coverage_min, unsure_flags)
    return DeviationResult(
        element_id=element_id,
        signed_distance=signed,
        face_index=nearest,
        metrics=metrics,
        status=status,
        tolerance=tolerance,
        coverage_per_face=coverages,
        unsure_flags=tuple(unsure_flags),
    )


def apply_metrics(
    scene: Scene, results: Iterable[DeviationResult] | Mapping[str, DeviationResult]
) -> None:
    """Write metrics and status into ``scene.derived["elements"][id]``.

    Only derived values change: element params, parameter origins and ``element.metrics`` are
    untouched, and ``recompute`` preserves these keys (it owns only quantities, geometry and
    issues). Keys written: ``metrics`` (plain dict), ``status`` (string), ``tolerance`` (m).
    """
    items = results.values() if isinstance(results, Mapping) else results
    derived = scene.derived.setdefault("elements", {})
    for result in items:
        if result.element_id not in scene.elements:
            raise KeyError(f"unknown element {result.element_id!r}")
        entry = derived.setdefault(result.element_id, {})
        entry["metrics"] = result.metrics.model_dump(mode="json")
        entry["status"] = result.status.value
        entry["tolerance"] = result.tolerance
        if result.unsure_flags:
            entry["unsure_flags"] = list(result.unsure_flags)
        else:
            entry.pop("unsure_flags", None)


def tolerance_colors(
    signed_distance: Array, tolerance: float, colors: ToleranceColors | None = None
) -> NDArray[np.uint8]:
    """RGB uint8 per point: within the tolerance, in front beyond it, behind beyond it.

    Within means ``|d| <= tolerance``; front is ``d > tolerance`` (positive side), behind is
    ``d < -tolerance``. NaN distances get neutral gray.
    """
    palette = (colors or ToleranceColors()).palette()
    dist = np.asarray(signed_distance, dtype=np.float64)
    rgb = np.empty((len(dist), 3), dtype=np.uint8)
    rgb[:] = palette[0]
    rgb[dist > tolerance] = palette[1]
    rgb[dist < -tolerance] = palette[2]
    rgb[np.isnan(dist)] = (128, 128, 128)
    return rgb


__all__ = [
    "DeviationResult",
    "DeviationSettings",
    "Face",
    "apply_metrics",
    "compute_deviation",
    "element_faces",
    "face_coverage",
    "signed_distances",
    "status_from_metrics",
    "tolerance_colors",
]
