"""Conforming, labelled terrain TIN construction.

The mesher deliberately keeps pavement boundaries as edges in one common mesh.
Downstream exporters can therefore omit pavement-labelled triangles instead of
performing a fragile boolean subtraction on a terrain mesh.
"""
from __future__ import annotations

import logging
import math
from collections.abc import Iterable

import numpy as np
import shapely
from scipy.spatial import Delaunay, cKDTree
from shapely.geometry import LineString, MultiLineString, Polygon
from shapely.ops import unary_union
from shapely.validation import make_valid

from samcloud.drone.contracts import LabeledTin, TIN_LABELS


LOGGER = logging.getLogger(__name__)
_EPS = 1e-9
_CONSTRAINT_MERGE_TOLERANCE = 2.5e-3


def _polygon_parts(value) -> list[Polygon]:
    """Return valid polygonal parts from a polygon-like input value."""
    if hasattr(value, "polygon"):
        value = value.polygon
    if value is None or value.is_empty:
        return []
    fixed = make_valid(value)
    if fixed.geom_type == "Polygon":
        return [fixed]
    if hasattr(fixed, "geoms"):
        return [part for geom in fixed.geoms for part in _polygon_parts(geom)]
    return []


def _clean_line(points: np.ndarray) -> np.ndarray:
    """Drop repeated XY coordinates while retaining the first supplied Z value."""
    values = np.asarray(points, dtype=float)
    if values.ndim != 2 or values.shape[1] < 2:
        return np.empty((0, 3), dtype=float)
    if values.shape[1] == 2:
        values = np.column_stack((values, np.full(len(values), np.nan)))
    else:
        values = values[:, :3].copy()
    if len(values) < 2:
        return values
    keep = np.r_[True, np.linalg.norm(np.diff(values[:, :2], axis=0), axis=1) > _EPS]
    return values[keep]


def _iter_raw_lines(regions: dict[str, list[Polygon]] | None, breaklines: Iterable[np.ndarray]) -> list[np.ndarray]:
    """Return every polygon ring and breakline as an open sequence of 3D points."""
    lines: list[np.ndarray] = []
    for polygons in (regions or {}).values():
        for value in polygons:
            for poly in _polygon_parts(value):
                rings = [poly.exterior, *poly.interiors]
                for ring in rings:
                    coords = np.asarray(ring.coords, dtype=float)
                    if len(coords) > 1:
                        # Keep the repeated first coordinate: unlike breaklines,
                        # polygon rings need their closing segment as a constraint.
                        lines.append(_clean_line(coords))
    for line in breaklines or ():
        if hasattr(line, "points"):
            line = line.points
        clean = _clean_line(np.asarray(line, dtype=float))
        if len(clean) >= 2:
            lines.append(clean)
    return [line for line in lines if len(line) >= 2]


def _merge_constraint_vertices(lines: list[np.ndarray], tolerance: float = _CONSTRAINT_MERGE_TOLERANCE) -> list[np.ndarray]:
    """Merge nearly coincident constraint vertices while preserving breakline Z values."""
    if not lines:
        return []
    lengths = [len(line) for line in lines]
    points = np.vstack(lines)
    if len(points) < 2:
        return lines
    parents = np.arange(len(points))

    def find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = int(parents[index])
        return index

    def join(first: int, second: int) -> None:
        first, second = find(first), find(second)
        if first != second:
            parents[second] = first

    for first, second in cKDTree(points[:, :2]).query_pairs(tolerance):
        join(int(first), int(second))
    groups: dict[int, list[int]] = {}
    for index in range(len(points)):
        groups.setdefault(find(index), []).append(index)
    merged = points.copy()
    for members in groups.values():
        # Later input lines are breaklines, whose finite elevation takes priority.
        preferred = next((index for index in reversed(members) if np.isfinite(points[index, 2])), members[0])
        merged[members] = points[preferred]
    result = []
    start = 0
    for length in lengths:
        result.append(_clean_line(merged[start:start + length]))
        start += length
    return [line for line in result if len(line) >= 2]


def _node_constraint_lines(lines: list[np.ndarray]) -> list[np.ndarray]:
    """Split constraint linework at crossings and shared/overlapping endpoints.

    A Delaunay mesh cannot contain two partially overlapping edges with different
    endpoints.  Noding before densification gives every overlapping constraint
    the same atomic segments and makes crossings explicit vertices.
    """
    if not lines:
        return []
    linework = unary_union([LineString(line[:, :2]) for line in lines])
    geometries = [linework] if linework.geom_type == "LineString" else list(getattr(linework, "geoms", ()))

    def elevation(xy: np.ndarray) -> float:
        for line in reversed(lines):
            for start, end in zip(line[:-1], line[1:]):
                direction = end[:2] - start[:2]
                squared_length = float(np.dot(direction, direction))
                if squared_length <= _EPS or not (np.isfinite(start[2]) or np.isfinite(end[2])):
                    continue
                fraction = float(np.clip(np.dot(xy - start[:2], direction) / squared_length, 0.0, 1.0))
                closest = start[:2] + fraction * direction
                if np.linalg.norm(xy - closest) <= _CONSTRAINT_MERGE_TOLERANCE:
                    start_z = start[2] if np.isfinite(start[2]) else end[2]
                    end_z = end[2] if np.isfinite(end[2]) else start_z
                    return float(start_z * (1.0 - fraction) + end_z * fraction)
        return float("nan")

    result: list[np.ndarray] = []
    for geometry in geometries:
        if geometry.geom_type != "LineString":
            continue
        xy = np.asarray(geometry.coords, dtype=float)[:, :2]
        if len(xy) >= 2:
            result.append(_clean_line(np.column_stack((xy, [elevation(point) for point in xy]))))
    return [line for line in result if len(line) >= 2]


def _prepare_constraint_lines(raw_lines: list[np.ndarray], point_tree: cKDTree, fallback_spacing: float) -> list[np.ndarray]:
    """Normalize, node, and densify the linework used as TIN constraints."""
    return _densify_lines(_node_constraint_lines(_merge_constraint_vertices(raw_lines)), point_tree, fallback_spacing)


def _nearest_spacing(xy: np.ndarray) -> tuple[cKDTree, float]:
    tree = cKDTree(xy)
    if len(xy) < 2:
        return tree, 1.0
    distances = tree.query(xy, k=2, workers=-1)[0][:, 1]
    valid = distances[np.isfinite(distances) & (distances > _EPS)]
    return tree, float(np.median(valid)) if len(valid) else 1.0


def _densify_lines(lines: list[np.ndarray], point_tree: cKDTree, fallback_spacing: float) -> list[np.ndarray]:
    """Densify lines to half their local DTM spacing, preserving breakline Z values."""
    output: list[np.ndarray] = []
    for line in lines:
        dense: list[np.ndarray] = [line[0]]
        for start, end in zip(line[:-1], line[1:]):
            length = float(np.linalg.norm(end[:2] - start[:2]))
            if length <= _EPS:
                continue
            midpoint = (start[:2] + end[:2]) * 0.5
            if point_tree.n:
                distances = np.atleast_1d(point_tree.query(midpoint, k=min(2, point_tree.n))[0])
                positive = distances[distances > _EPS]
                local = float(positive[0]) if len(positive) else fallback_spacing
            else:
                local = fallback_spacing
            local = max(local, fallback_spacing * 0.1, _EPS)
            pieces = max(1, int(math.ceil(length / (0.5 * local))))
            fractions = np.linspace(0.0, 1.0, pieces + 1)[1:]
            for frac in fractions:
                p = start * (1.0 - frac) + end * frac
                dense.append(p)
        output.append(np.asarray(dense, dtype=float))
    return output


def _segments(lines: list[np.ndarray]) -> list[tuple[np.ndarray, np.ndarray, int]]:
    return [(a.copy(), b.copy(), root) for root, line in enumerate(lines) for a, b in zip(line[:-1], line[1:])
            if np.linalg.norm(a[:2] - b[:2]) > _EPS]


def _segments_from_value(constraints) -> list[tuple[np.ndarray, np.ndarray, int]]:
    """Accept the convenient public constraint forms used by callers and QA code."""
    if constraints is None:
        return []
    if isinstance(constraints, dict):
        if "regions" in constraints or "breaklines" in constraints:
            lines = _iter_raw_lines(constraints.get("regions", {}), constraints.get("breaklines", ()))
        else:
            lines = _iter_raw_lines(constraints, ())
        return _segments(lines)
    lines = []
    for item in constraints:
        arr = np.asarray(item, dtype=float)
        if arr.ndim == 2 and arr.shape[1] >= 2:
            lines.append(_clean_line(arr))
    return _segments(lines)


def _interpolate_z(points_xy: np.ndarray, ground: np.ndarray, sampler) -> np.ndarray:
    if sampler is not None:
        return np.asarray([float(sampler(float(x), float(y))) for x, y in points_xy], dtype=float)
    if len(ground) < 3:
        return np.full(len(points_xy), float(ground[0, 2]) if len(ground) else 0.0)
    try:
        tri = Delaunay(ground[:, :2])
        simplex = tri.find_simplex(points_xy)
        z = np.empty(len(points_xy), dtype=float)
        valid = simplex >= 0
        transform = tri.transform[simplex[valid], :2]
        delta = points_xy[valid] - tri.transform[simplex[valid], 2]
        weights = np.einsum("nij,nj->ni", transform, delta)
        barycentric = np.column_stack((weights, 1.0 - weights.sum(axis=1)))
        z[valid] = np.einsum("ni,ni->n", barycentric, ground[tri.simplices[simplex[valid]], 2])
        if (~valid).any():
            tree = cKDTree(ground[:, :2])
            nearest = tree.query(points_xy[~valid])[1]
            z[~valid] = ground[nearest, 2]
        return z
    except Exception:
        tree = cKDTree(ground[:, :2])
        return ground[tree.query(points_xy)[1], 2]


def _remove_near_constraints(ground: np.ndarray, lines: list[np.ndarray], buffer: float) -> np.ndarray:
    if buffer <= 0 or not lines or not len(ground):
        return ground
    geometry = MultiLineString([line[:, :2] for line in lines if len(line) >= 2])
    if geometry.is_empty:
        return ground
    points = shapely.points(ground[:, 0], ground[:, 1])
    keep = ~shapely.dwithin(points, geometry, buffer)
    return ground[np.asarray(keep, dtype=bool)]


def _unique_vertices(ground: np.ndarray, segments, sampler) -> tuple[np.ndarray, list[tuple[int, int, int]]]:
    """Merge all input sources by XY, giving explicit breakline Z precedence."""
    records: list[np.ndarray] = []
    lookup: dict[tuple[int, int], int] = {}
    scale = 1e9

    def add(point: np.ndarray, prefer: bool = False) -> int:
        key = tuple(np.rint(point[:2] * scale).astype(np.int64))
        idx = lookup.get(key)
        if idx is None:
            lookup[key] = len(records)
            records.append(point.copy())
            return len(records) - 1
        if prefer and np.isfinite(point[2]):
            records[idx][2] = point[2]
        return idx

    for start, end, _ in segments:
        add(start, prefer=True)
        add(end, prefer=True)
    for point in ground:
        add(point, prefer=False)
    vertices = np.asarray(records, dtype=float)
    missing_z = ~np.isfinite(vertices[:, 2])
    if missing_z.any():
        vertices[missing_z, 2] = _interpolate_z(vertices[missing_z, :2], ground, sampler)
    indexed = []
    for start, end, root in segments:
        i = add(start, prefer=True)
        j = add(end, prefer=True)
        if i != j:
            indexed.append((i, j, root))
    return vertices, indexed


def _edge_set(triangles: np.ndarray) -> set[tuple[int, int]]:
    if len(triangles) == 0:
        return set()
    pairs = np.vstack((triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]]))
    pairs.sort(axis=1)
    return {tuple(edge) for edge in pairs}


def _triangulate(vertices: np.ndarray) -> np.ndarray:
    if len(vertices) < 3:
        return np.empty((0, 3), dtype=np.int64)
    try:
        return Delaunay(vertices[:, :2]).simplices.astype(np.int64, copy=False)
    except Exception as exc:
        raise ValueError("DTM support points do not span a 2D triangulation") from exc


def _filter_triangles(vertices: np.ndarray, triangles: np.ndarray, protected_edges: set[tuple[int, int]] | None = None) -> np.ndarray:
    if not len(triangles):
        return triangles
    xy = vertices[:, :2]
    a, b, c = xy[triangles[:, 0]], xy[triangles[:, 1]], xy[triangles[:, 2]]
    ab = b - a
    ac = c - a
    twice_area = np.abs(ab[:, 0] * ac[:, 1] - ab[:, 1] * ac[:, 0])
    edges = np.column_stack((np.linalg.norm(a - b, axis=1), np.linalg.norm(b - c, axis=1), np.linalg.norm(c - a, axis=1)))
    with np.errstate(divide="ignore", invalid="ignore"):
        radius = edges.prod(axis=1) / (2.0 * twice_area)
    median_edge = float(np.median(edges)) if len(edges) else 0.0
    keep = (twice_area * 0.5 >= 1e-8) & np.isfinite(radius)
    if median_edge > 0:
        keep &= radius <= 5.0 * median_edge
    if protected_edges:
        tri_edges = np.stack((triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]]), axis=1)
        tri_edges.sort(axis=2)
        touches_constraint = np.array(
            [any(tuple(edge) in protected_edges for edge in tri_edges[index]) for index in range(len(triangles))], dtype=bool
        )
        keep |= touches_constraint
    return triangles[keep]


def _label_triangles(vertices: np.ndarray, triangles: np.ndarray, regions: dict[str, list[Polygon]]) -> np.ndarray:
    labels = np.full(len(triangles), TIN_LABELS["terrain"], dtype=np.uint8)
    if not len(triangles):
        return labels
    centres = vertices[triangles, :2].mean(axis=1)
    # Earlier assignments win, matching the agreed pavement priority.
    for name in ("road", "sidewalk", "path", "lawn"):
        polygons = [part for poly in regions.get(name, []) for part in _polygon_parts(poly)]
        if not polygons:
            continue
        geom = unary_union(polygons)
        inside = np.asarray(shapely.contains_xy(geom, centres[:, 0], centres[:, 1]), dtype=bool)
        labels[(labels == TIN_LABELS["terrain"]) & inside] = TIN_LABELS[name]
    return labels


def build_labeled_tin(
    ground_xy_z,
    *,
    regions: dict[str, list[Polygon]],
    breaklines: list[np.ndarray] = (),
    dtm_sampler=None,
    buffer: float = 0.05,
    min_angle_deg: float = 20,
    max_steiner_per_edge: int = 50,
    thin_spacing: float | None = None,
) -> LabeledTin:
    """Build a common terrain/pavement TIN with recovered polygon and curb edges.

    Constraints are densified before Delaunay triangulation.  Missing edges are
    recovered by midpoint Steiner insertion and global retriangulation; this is
    predictable and robust for the modest number of pavement boundaries typical
    of one drone survey.
    """
    ground = np.asarray(ground_xy_z, dtype=float)
    if ground.ndim != 2 or ground.shape[1] != 3 or len(ground) < 3:
        raise ValueError("ground_xy_z must be an (N, 3) array with at least three points")
    ground = ground[np.isfinite(ground).all(axis=1)]
    if thin_spacing is not None and thin_spacing > 0:
        bins = np.floor(ground[:, :2] / float(thin_spacing)).astype(np.int64)
        _, first = np.unique(bins, axis=0, return_index=True)
        ground = ground[np.sort(first)]
    point_tree, median_spacing = _nearest_spacing(ground[:, :2])
    raw_lines = _iter_raw_lines(regions, breaklines)
    lines = _prepare_constraint_lines(raw_lines, point_tree, median_spacing)
    segments = _segments(lines)
    retained_ground = _remove_near_constraints(ground, lines, float(buffer))
    if len(retained_ground) < 3:
        retained_ground = ground

    # Each refinement adds at most the requested number of points per original
    # densified edge.  Most valid input needs no recovery iterations at all.
    steiner_counts: dict[int, int] = {}
    unresolved: list[tuple[np.ndarray, np.ndarray, int]] = []
    for _ in range(max(0, int(max_steiner_per_edge)) + 1):
        vertices, indexed = _unique_vertices(retained_ground, segments, dtm_sampler)
        triangles = _triangulate(vertices)
        edges = _edge_set(triangles)
        missing = [(start, end, root) for (start, end, root), (i, j, _) in zip(segments, indexed)
                   if tuple(sorted((i, j))) not in edges]
        if not missing:
            unresolved = []
            break
        unresolved = missing
        replacement: list[tuple[np.ndarray, np.ndarray, int]] = []
        split_keys = {
            (tuple(np.rint(item[0][:2] * 1e9).astype(np.int64)), tuple(np.rint(item[1][:2] * 1e9).astype(np.int64)), item[2])
            for item in missing if steiner_counts.get(item[2], 0) < max_steiner_per_edge
        }
        for segment in segments:
            segment_key = (
                tuple(np.rint(segment[0][:2] * 1e9).astype(np.int64)),
                tuple(np.rint(segment[1][:2] * 1e9).astype(np.int64)), segment[2],
            )
            if segment_key not in split_keys:
                replacement.append(segment)
                continue
            start, end, root = segment
            middle = (start + end) * 0.5
            replacement.extend(((start, middle, root), (middle, end, root)))
            steiner_counts[root] = steiner_counts.get(root, 0) + 1
        if len(replacement) == len(segments):
            break
        segments = replacement
    if unresolved:
        # The final refinement can have replaced a segment after the last
        # iteration.  Triangulate it once more so warning state and metadata
        # always describe the same mesh.
        vertices, indexed = _unique_vertices(retained_ground, segments, dtm_sampler)
        triangles = _triangulate(vertices)
        edges = _edge_set(triangles)
        unresolved = [(start, end, root) for (start, end, root), (i, j, _) in zip(segments, indexed)
                      if tuple(sorted((i, j))) not in edges]
    if unresolved:
        LOGGER.warning("could not recover %d constrained TIN edge(s)", len(unresolved))

    constrained_edges = {tuple(sorted((i, j))) for i, j, _ in indexed}
    triangles = _filter_triangles(vertices, triangles, constrained_edges)
    labels = _label_triangles(vertices, triangles, regions)
    tin = LabeledTin(vertices=vertices, triangles=triangles, labels=labels)
    # Kept as non-contract metadata so QA can inspect the exact densified edges.
    tin.constraint_segments = [np.vstack((start, end)) for start, end, _ in segments]
    tin.min_angle_deg = float(min_angle_deg)
    return tin


def missing_constraint_edges(tin: LabeledTin, constraints=None) -> list:
    """Return constrained segments that are absent from ``tin`` within 1 mm."""
    raw = getattr(tin, "constraint_segments", ()) if constraints is None or isinstance(constraints, dict) else constraints
    segments = _segments_from_value(raw)
    if constraints is None:
        # Metadata stores individual two-point segments, which need no ring parsing.
        segments = _segments_from_value(raw)
    if not segments:
        return []
    vertices = np.asarray(tin.vertices, dtype=float)
    tree = cKDTree(vertices[:, :2])
    edges = _edge_set(np.asarray(tin.triangles, dtype=np.int64))
    missing = []
    for start, end, _ in segments:
        distance, indices = tree.query(np.vstack((start[:2], end[:2])), distance_upper_bound=1e-3)
        if np.any(~np.isfinite(distance)) or np.any(indices >= len(vertices)):
            missing.append((start[:2].copy(), end[:2].copy()))
            continue
        if tuple(sorted(map(int, indices))) not in edges:
            missing.append((start[:2].copy(), end[:2].copy()))
    return missing


def tin_quality(tin: LabeledTin) -> dict:
    """Compute compact geometric and label QA statistics for a labelled TIN."""
    triangles = np.asarray(tin.triangles, dtype=np.int64)
    vertices = np.asarray(tin.vertices, dtype=float)
    label_counts = {name: int((tin.labels == code).sum()) for name, code in TIN_LABELS.items()}
    label_areas = {name: 0.0 for name in TIN_LABELS}
    if not len(triangles):
        return {"triangle_count": 0, "min_angle_p1_deg": 0.0, "min_angle_median_deg": 0.0,
                "share_below_min_angle": 0.0, "degenerate_count": 0, "label_counts": label_counts,
                "label_areas": label_areas}
    xyz = vertices[triangles]
    lengths = np.column_stack((np.linalg.norm(xyz[:, 1] - xyz[:, 2], axis=1),
                               np.linalg.norm(xyz[:, 0] - xyz[:, 2], axis=1),
                               np.linalg.norm(xyz[:, 0] - xyz[:, 1], axis=1)))
    with np.errstate(invalid="ignore", divide="ignore"):
        angles = np.degrees(np.arccos(np.clip((lengths[:, [1, 2, 0]] ** 2 + lengths[:, [2, 0, 1]] ** 2 - lengths ** 2)
                                               / (2.0 * lengths[:, [1, 2, 0]] * lengths[:, [2, 0, 1]]), -1.0, 1.0)))
    minimum = np.nanmin(angles, axis=1)
    ab = xyz[:, 1, :2] - xyz[:, 0, :2]
    ac = xyz[:, 2, :2] - xyz[:, 0, :2]
    areas = 0.5 * np.abs(ab[:, 0] * ac[:, 1] - ab[:, 1] * ac[:, 0])
    for name, code in TIN_LABELS.items():
        label_areas[name] = float(areas[tin.labels == code].sum())
    threshold = float(getattr(tin, "min_angle_deg", 20.0))
    return {
        "triangle_count": int(len(triangles)),
        "min_angle_p1_deg": float(np.nanpercentile(minimum, 1)),
        "min_angle_median_deg": float(np.nanmedian(minimum)),
        "share_below_min_angle": float(np.mean(minimum < threshold)),
        "degenerate_count": int(np.sum(areas < 1e-8)),
        "label_counts": label_counts,
        "label_areas": label_areas,
    }


def label_boundary_consistent(tin: LabeledTin, regions: dict[str, list[Polygon]]) -> float:
    """Return the maximum label-boundary deviation from the supplied regions."""
    maximum = 0.0
    for name, code in TIN_LABELS.items():
        if name == "terrain" or not regions.get(name):
            continue
        target_parts = [part for poly in regions[name] for part in _polygon_parts(poly)]
        if not target_parts:
            continue
        selected = np.asarray(tin.triangles)[np.asarray(tin.labels) == code]
        if not len(selected):
            return float("inf")
        mesh = unary_union([Polygon(tin.vertices[tri, :2]) for tri in selected])
        target = unary_union(target_parts)
        maximum = max(maximum, float(mesh.boundary.hausdorff_distance(target.boundary)))
    return maximum


__all__ = ["build_labeled_tin", "missing_constraint_edges", "tin_quality", "label_boundary_consistent"]
