"""Conforming, labelled terrain TIN construction.

The mesher deliberately keeps pavement boundaries as edges in one common mesh.
Downstream exporters can therefore omit pavement-labelled triangles instead of
performing a fragile boolean subtraction on a terrain mesh.
"""
from __future__ import annotations

import logging
import math
from collections.abc import Callable, Iterable

import numpy as np
import shapely
from scipy.spatial import ConvexHull, Delaunay, cKDTree
from shapely.geometry import LineString, MultiLineString, Polygon
from shapely.ops import unary_union
from shapely.validation import make_valid

from samcloud.drone.contracts import LabeledTin, TIN_LABELS


LOGGER = logging.getLogger(__name__)
_EPS = 1e-9
_CONSTRAINT_MERGE_TOLERANCE = 1.0e-2
_RECOVERY_ROUNDS = 8  # bounded midpoint-split rounds for constraints that Delaunay does not contain


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
    # Near-coincident constraints (mm offsets between a curb line and a polygon edge) cannot coexist in a
    # Delaunay mesh: every line is snapped onto earlier lines that run within the merge tolerance.
    raw = [LineString(line[:, :2]) for line in lines]
    tree = shapely.STRtree(raw)
    snapped = list(raw)
    for index, geometry in enumerate(raw):
        near = [int(j) for j in tree.query(geometry, predicate="dwithin", distance=_CONSTRAINT_MERGE_TOLERANCE) if j < index]
        if near:
            reference = unary_union([snapped[j] for j in near])
            candidate = shapely.snap(geometry, reference, _CONSTRAINT_MERGE_TOLERANCE)
            if candidate.geom_type == "LineString" and not candidate.is_empty:
                snapped[index] = candidate
    # Vertices that now lie on a neighbouring segment must split it, or the noding below misses the contact.
    tree2 = shapely.STRtree(snapped)
    for index, geometry in enumerate(snapped):
        near = [int(j) for j in tree2.query(geometry, predicate="dwithin", distance=1e-5) if j != index]
        if near:
            points = shapely.multipoints(np.vstack([np.asarray(snapped[j].coords) for j in near]))
            candidate = shapely.snap(geometry, points, 1e-5)
            if candidate.geom_type == "LineString" and not candidate.is_empty:
                snapped[index] = candidate
    linework = unary_union(snapped)
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


def _barycentric_elevation(triangulation: Delaunay, z: np.ndarray, xy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Interpolate elevations and return the containing simplex for each XY point."""
    simplex = triangulation.find_simplex(xy)
    values = np.full(len(xy), np.nan, dtype=float)
    valid = simplex >= 0
    if valid.any():
        transform = triangulation.transform[simplex[valid], :2]
        delta = xy[valid] - triangulation.transform[simplex[valid], 2]
        weights = np.einsum("nij,nj->ni", transform, delta)
        barycentric = np.column_stack((weights, 1.0 - weights.sum(axis=1)))
        values[valid] = np.einsum("ni,ni->n", barycentric, z[triangulation.simplices[simplex[valid]]])
    return values, simplex


def _grid_reference(ground: np.ndarray, cell: float) -> np.ndarray:
    """Reduce ground points to one median-elevation point per regular grid cell."""
    xy = ground[:, :2]
    origin = xy.min(axis=0)
    bins = np.floor((xy - origin) / cell).astype(np.int64)
    unique_bins, inverse = np.unique(bins, axis=0, return_inverse=True)
    # Sorting by cell then elevation permits a fully vectorised grouped median.
    order = np.lexsort((ground[:, 2], inverse))
    counts = np.bincount(inverse, minlength=len(unique_bins))
    starts = np.r_[0, np.cumsum(counts[:-1])]
    lower = ground[order[starts + (counts - 1) // 2], 2]
    upper = ground[order[starts + counts // 2], 2]
    centres = origin + (unique_bins.astype(float) + 0.5) * cell
    return np.column_stack((centres, (lower + upper) * 0.5))


def _initial_support(reference: np.ndarray, spacing: float, max_vertices: int) -> np.ndarray:
    """Select one nearby reference point per coarse grid cell plus hull corners."""
    xy = reference[:, :2]
    origin = xy.min(axis=0)
    bins = np.floor((xy - origin) / spacing).astype(np.int64)
    unique_bins, inverse = np.unique(bins, axis=0, return_inverse=True)
    centres = origin + (unique_bins.astype(float) + 0.5) * spacing
    distances = np.einsum("ij,ij->i", xy - centres[inverse], xy - centres[inverse])
    order = np.lexsort((distances, inverse))
    first = np.r_[True, inverse[order][1:] != inverse[order][:-1]]
    coarse = order[first]
    try:
        hull = ConvexHull(xy).vertices
    except Exception:
        hull = np.empty(0, dtype=np.int64)
    # Hull vertices establish coverage; the coarse representatives fill the cap.
    selected = np.r_[hull, coarse]
    _, first_positions = np.unique(selected, return_index=True)
    selected = selected[np.sort(first_positions)]
    return selected[:max_vertices]


def select_tin_support(
    ground,
    *,
    dtm_sampler=None,
    tolerance: float,
    initial_spacing: float,
    max_vertices: int,
    constraints_xy=None,
    max_rounds: int = 4,
    tolerance_fn=None,
) -> tuple[np.ndarray, dict]:
    """Select bounded TIN support points with batched vertical-error refinement.

    A median grid forms a noise-resistant reference surface.  Each refinement
    round adds the largest violating reference point in every TIN triangle.  ``tolerance_fn``
    (XY array -> tolerance array) makes the tolerance position dependent, e.g. tighter on pavements.
    """
    points = np.asarray(ground, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("ground must be an (N, 3) array")
    points = points[np.isfinite(points).all(axis=1)]
    if len(points) < 3:
        raise ValueError("ground must contain at least three finite points")
    if tolerance <= 0 or initial_spacing <= 0 or max_vertices < 3:
        raise ValueError("tolerance, initial_spacing, and max_vertices must be positive")

    _, median_spacing = _nearest_spacing(points[:, :2])
    reference = _grid_reference(points, max(float(median_spacing), 0.1))
    initial = _initial_support(reference, float(initial_spacing), int(max_vertices))
    selected = reference[initial]
    ref_tol = np.full(len(reference), float(tolerance)) if tolerance_fn is None else np.asarray(tolerance_fn(reference[:, :2]), dtype=float)

    if constraints_xy is not None and len(selected) < max_vertices:
        constraint_xy = np.asarray(constraints_xy, dtype=float)
        if constraint_xy.ndim == 2 and constraint_xy.shape[1] >= 2 and len(constraint_xy):
            constraint_xy = constraint_xy[:, :2]
            z = _interpolate_z(constraint_xy, points, dtm_sampler)
            extra = np.column_stack((constraint_xy, z))
            # Constraints are non-negotiable topology; retain them before
            # optional coarse support when applying the vertex budget.
            combined = np.vstack((extra, selected))
            keys = np.rint(combined[:, :2] * 1e9).astype(np.int64)
            _, first = np.unique(keys, axis=0, return_index=True)
            selected = combined[np.sort(first)[:max_vertices]]

    rounds = 0
    round_additions: list[int] = []
    round_triangle_counts: list[int] = []
    errors = np.empty(0, dtype=float)
    stop_reason = "max_rounds"
    for _ in range(max(1, int(max_rounds))):
        triangulation = Delaunay(selected[:, :2])
        interpolated, simplex = _barycentric_elevation(triangulation, selected[:, 2], reference[:, :2])
        valid = np.isfinite(interpolated)
        errors = np.abs(reference[valid, 2] - interpolated[valid])
        limit = ref_tol[valid]
        if not len(errors):
            stop_reason = "no_interpolable_candidates"
            break
        if bool((errors <= limit).all()):
            stop_reason = "tolerance_reached"
            break
        if len(selected) >= max_vertices:
            stop_reason = "max_vertices"
            break

        valid_indices = np.flatnonzero(valid)
        valid_simplex = simplex[valid]
        # Lexicographic sort yields the largest error at the start of each simplex group.
        excess = errors - limit
        order = np.lexsort((-excess, valid_simplex))
        ordered_simplex = valid_simplex[order]
        first = np.r_[True, ordered_simplex[1:] != ordered_simplex[:-1]]
        candidates = valid_indices[order[first][excess[order[first]] > 0]]
        if not len(candidates):
            stop_reason = "no_violating_candidates"
            break
        remaining = int(max_vertices) - len(selected)
        if len(candidates) > remaining:
            candidates = candidates[np.argsort(excess[order[first]][excess[order[first]] > 0])[::-1][:remaining]]
        round_triangle_counts.append(int(len(triangulation.simplices)))
        selected = np.vstack((selected, reference[candidates]))
        rounds += 1
        round_additions.append(int(len(candidates)))

    triangulation = Delaunay(selected[:, :2])
    interpolated, _ = _barycentric_elevation(triangulation, selected[:, 2], reference[:, :2])
    errors = np.abs(reference[np.isfinite(interpolated), 2] - interpolated[np.isfinite(interpolated)])
    final_limit = ref_tol[np.isfinite(interpolated)]
    report = {
        "rounds": int(rounds),
        "vertices": int(len(selected)),
        "max_error_m": float(errors.max()) if len(errors) else 0.0,
        "p95_error_m": float(np.percentile(errors, 95)) if len(errors) else 0.0,
        "rms_error_m": float(np.sqrt(np.mean(errors ** 2))) if len(errors) else 0.0,
        "candidate_cells": int(len(reference)),
        "round_additions": round_additions,
        "round_triangle_counts": round_triangle_counts,
        "max_added_per_triangle_per_round": 1,
        "stop_reason": stop_reason,
        "violations": int((errors > final_limit).sum()),
    }
    return selected, report


def _densify_lines(lines: list[np.ndarray], point_tree: cKDTree, fallback_spacing: float) -> list[np.ndarray]:
    """Densify constraints once at a bounded DTM-scale spacing."""
    output: list[np.ndarray] = []
    for line in lines:
        dense: list[np.ndarray] = [line[0]]
        for start, end in zip(line[:-1], line[1:]):
            length = float(np.linalg.norm(end[:2] - start[:2]))
            if length <= _EPS:
                continue
            # A fixed spacing avoids a nearest-neighbour query and a Python
            # loop for every dense DTM cell.  It also bounds line vertices on
            # long surveyed boundaries while retaining curb breaks.
            spacing = float(np.clip(2.0 * float(fallback_spacing), 0.2, 1.0))
            pieces = max(1, int(math.ceil(length / spacing)))
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
        values = np.asarray(points_xy, dtype=float).reshape(-1, 2)
        sampled = np.asarray(sampler(values), dtype=float).reshape(-1)
        if len(sampled) != len(values):
            raise ValueError("DTM sampler returned an unexpected number of elevations")
        return sampled
    if len(ground) < 3:
        return np.full(len(points_xy), float(ground[0, 2]) if len(ground) else 0.0)
    # Constraint elevation is sampled once.  A tree lookup is substantially
    # cheaper than constructing a second full Delaunay triangulation merely to
    # obtain these values, and is robust at raster boundaries.
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
    scale = 1e9
    segment_count = len(segments)
    segment_points = np.empty((segment_count * 2, 3), dtype=float)
    if segment_count:
        segment_points[0::2] = np.asarray([start for start, _, _ in segments], dtype=float)
        segment_points[1::2] = np.asarray([end for _, end, _ in segments], dtype=float)
    records = np.vstack((segment_points, ground))
    keys = np.rint(records[:, :2] * scale).astype(np.int64)
    _, first, inverse = np.unique(keys, axis=0, return_index=True, return_inverse=True)
    record_order = np.argsort(first, kind="stable")
    output_index = np.empty(len(first), dtype=np.int64)
    output_index[record_order] = np.arange(len(first))
    vertices = records[first[record_order]].copy()

    # The old insertion order means the last finite constraint elevation wins.
    if segment_count:
        segment_inverse = inverse[:len(segment_points)]
        finite = np.isfinite(segment_points[:, 2])
        last_preferred = np.full(len(first), -1, dtype=np.int64)
        np.maximum.at(last_preferred, segment_inverse[finite], np.flatnonzero(finite))
        preferred_groups = np.flatnonzero(last_preferred >= 0)
        vertices[output_index[preferred_groups], 2] = segment_points[last_preferred[preferred_groups], 2]
    missing_z = ~np.isfinite(vertices[:, 2])
    if missing_z.any():
        vertices[missing_z, 2] = _interpolate_z(vertices[missing_z, :2], ground, sampler)
    indexed = []
    if segment_count:
        endpoint_indices = output_index[inverse[:len(segment_points)]].reshape(-1, 2)
        indexed = [(int(i), int(j), root) for (i, j), (_, _, root) in zip(endpoint_indices, segments) if i != j]
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
        protected = np.asarray(sorted(protected_edges), dtype=triangles.dtype)
        edge_dtype = np.dtype([("first", triangles.dtype), ("second", triangles.dtype)])
        tri_keys = np.ascontiguousarray(tri_edges.reshape(-1, 2)).view(edge_dtype).reshape(-1)
        protected_keys = np.ascontiguousarray(protected).view(edge_dtype).reshape(-1)
        touches_constraint = np.isin(tri_keys, protected_keys).reshape(-1, 3).any(axis=1)
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


def _label_tolerance(regions: dict[str, list[Polygon]], default: float, by_label: dict[str, float]):
    """Position dependent tolerance: the tolerance of the region label containing a point (default elsewhere)."""
    areas: list[tuple[float, object]] = []
    for name in ("road", "sidewalk", "path", "lawn"):
        if name not in by_label:
            continue
        parts = [part for poly in regions.get(name, []) for part in _polygon_parts(poly)]
        if parts:
            merged = unary_union(parts)
            shapely.prepare(merged)
            areas.append((float(by_label[name]), merged))
    base = float(by_label.get("terrain", default))

    def tolerance_at(xy: np.ndarray) -> np.ndarray:
        out = np.full(len(xy), base, dtype=float)
        # earlier regions win, matching the labelling priority; applied in reverse so they overwrite
        for value, geom in reversed(areas):
            out[np.asarray(shapely.contains_xy(geom, xy[:, 0], xy[:, 1]), dtype=bool)] = value
        return out

    return tolerance_at


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
    tolerance: float | None = None,
    initial_spacing: float | None = None,
    max_vertices: int | None = None,
    max_refinement_rounds: int = 4,
    tolerance_by_label: dict[str, float] | None = None,
    manholes: Iterable = (),
    manhole_ring_segments: int = 16,
    manhole_below_m: float = 0.001,
    steps: Iterable | None = None,
    step_config: dict | None = None,
    flush=None,
) -> LabeledTin:
    """Build a common terrain/pavement TIN with recovered polygon and curb edges.

    Constraints are densified before Delaunay triangulation.  Missing edges are
    recovered by midpoint Steiner insertion and global retriangulation; this is
    predictable and robust for the modest number of pavement boundaries typical
    of one drone survey.

    ``manholes`` is a sequence of ``(x, y, cover_z, diameter)``: every manhole becomes a polygon ring constraint
    (``manhole_ring_segments`` corners, circumscribed around the cover) and each TIN vertex on or inside the ring is
    set to ``cover_z - manhole_below_m`` (all labels), so the terrain never covers a manhole.  ``steps`` are
    :class:`samcloud.terrain.steps.StepRun` objects: their top edge is added as breakline and the finished TIN is split
    along them (top edge at the upper height, bottom edge ``offset_m`` outside at the lower height).  ``flush`` is a
    :class:`samcloud.terrain.flush.FlushPlan`: its edge lines (finite z = pavement surface at the edge) are breaklines and the
    terrain-only vertices within its blend band are moved towards the edge height, so terrain and pavement meet flush.
    """
    breaklines = list(breaklines or ())
    steps = list(steps or ())
    manhole_rings = _manhole_rings(manholes, int(manhole_ring_segments), float(manhole_below_m))
    breaklines.extend(ring for ring, _centre, _radius in manhole_rings)
    breaklines.extend(run.top_line() for run in steps)
    if flush is not None:
        breaklines.extend(flush.lines)
    ground = np.asarray(ground_xy_z, dtype=float)
    if ground.ndim != 2 or ground.shape[1] != 3 or len(ground) < 3:
        raise ValueError("ground_xy_z must be an (N, 3) array with at least three points")
    ground = ground[np.isfinite(ground).all(axis=1)]
    # Constraint spacing must be measured against the unsimplified survey.  Measuring it
    # after error-driven support selection turns dense, nearby curb/pavement lines into
    # sparse parallel constraints that Delaunay cannot represent reliably.
    raw_lines = _iter_raw_lines(regions, breaklines)
    source_tree, source_spacing = _nearest_spacing(ground[:, :2])
    prepared_lines = _prepare_constraint_lines(raw_lines, source_tree, source_spacing)
    constraint_xy = np.vstack([line[:, :2] for line in prepared_lines]) if prepared_lines else None
    # The candidate surface is strictly the DTM cells.  A clearance comparable
    # to the cell spacing prevents ordinary support vertices from cutting
    # across a densified curb or polygon boundary.
    # Every constraint segment must be a Gabriel edge (no support vertex inside its diametral
    # circle), otherwise Delaunay does not contain it: clear half of the longest segment.
    longest = max((float(np.linalg.norm(b[:2] - a[:2])) for line in prepared_lines for a, b in zip(line[:-1], line[1:])), default=0.0)
    clearance = max(float(buffer), float(source_spacing) * 1.01, 0.505 * longest)
    candidate_ground = _remove_near_constraints(ground, prepared_lines, clearance)
    if len(candidate_ground) < 3:
        candidate_ground = ground
    simplify_report = None
    tolerance_fn = None
    if tolerance is not None and tolerance_by_label:
        tolerance_fn = _label_tolerance(regions, float(tolerance), tolerance_by_label)
    if tolerance is not None:
        ground, simplify_report = select_tin_support(
            candidate_ground,
            dtm_sampler=dtm_sampler,
            tolerance=float(tolerance),
            initial_spacing=float(initial_spacing if initial_spacing is not None else 2.0),
            max_vertices=int(max_vertices if max_vertices is not None else 200000),
            constraints_xy=constraint_xy,
            max_rounds=max_refinement_rounds,
            tolerance_fn=tolerance_fn,
        )
    else:
        ground = candidate_ground
    if thin_spacing is not None and thin_spacing > 0:
        bins = np.floor(ground[:, :2] / float(thin_spacing)).astype(np.int64)
        _, first = np.unique(bins, axis=0, return_index=True)
        ground = ground[np.sort(first)]
    # ``prepared_lines`` deliberately keeps the original survey-driven spacing even
    # when the terrain support is sparse; exact line vertices are merged below.
    lines = prepared_lines
    segments = _segments(lines)
    retained_ground = ground

    # Each refinement adds at most the requested number of points per original
    # densified edge.  Most valid input needs no recovery iterations at all.
    steiner_counts: dict[int, int] = {}
    unresolved: list[tuple[np.ndarray, np.ndarray, int]] = []
    # Recovery is bounded: midpoint splitting continues only while it reduces the number of missing
    # edges, and the best state is kept (splitting can make mm-scale tangles worse).
    best: tuple[int, list] | None = None
    previous: int | None = None
    for _ in range(min(_RECOVERY_ROUNDS, max(0, int(max_steiner_per_edge))) + 1):
        vertices, indexed = _unique_vertices(retained_ground, segments, dtm_sampler)
        triangles = _triangulate(vertices)
        edges = _edge_set(triangles)
        missing = [(start, end, root) for (start, end, root), (i, j, _) in zip(segments, indexed)
                   if tuple(sorted((i, j))) not in edges]
        if best is None or len(missing) < best[0]:
            best = (len(missing), segments)
        if not missing:
            unresolved = []
            break
        unresolved = missing
        if previous is not None and len(missing) >= previous:
            break
        previous = len(missing)
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
    if best is not None:
        segments = best[1]
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
    flush_report = None
    if flush is not None and len(flush.edge_z):
        from samcloud.terrain.flush import blend_terrain

        blended = LabeledTin(vertices=vertices, triangles=triangles, labels=labels)
        flush_report = blend_terrain(blended, flush)
        vertices = blended.vertices
    if manhole_rings:
        vertices = vertices.copy()
        for ring, centre, radius in manhole_rings:
            inside = np.hypot(vertices[:, 0] - centre[0], vertices[:, 1] - centre[1]) <= radius + 1e-6
            vertices[inside, 2] = ring[0, 2]
    tin = LabeledTin(vertices=vertices, triangles=triangles, labels=labels)
    # Kept as non-contract metadata so QA can inspect the exact densified edges.
    tin.constraint_segments = [np.vstack((start, end)) for start, end, _ in segments]
    tin.constraint_report = {"unrecovered_edges": int(len(unresolved)), "recovery_round_limit": _RECOVERY_ROUNDS}
    tin.min_angle_deg = float(min_angle_deg)
    if simplify_report is not None:
        tin.simplify_report = simplify_report
    tin.manhole_count = len(manhole_rings)
    if flush is not None:
        tin.flush_report = {**flush.report, **(flush_report or {})}
    if steps:
        from samcloud.terrain.steps import apply_steps

        apply_steps(tin, steps, {"terrain": {"step": step_config or {}}})
    return tin


def tin_sampler(tin) -> Callable:
    """Surface height sampler ``f(xy (N,2)) -> z (N,)`` of a (labelled) TIN, NaN outside.

    Where several triangles cover a point (the 1 cm step faces of :mod:`samcloud.terrain.steps` overlap their upper and
    lower neighbours in plan at the shared edge) the highest surface wins, so the sampler returns the UPPER side at a
    step.  Curbs, gutters, fences, walls and trees sample this one surface.
    """
    vertices = np.asarray(tin.vertices, dtype=np.float64)
    triangles = np.asarray(tin.triangles, dtype=np.int64)
    tree = shapely.STRtree(shapely.polygons(vertices[triangles][:, :, :2])) if len(triangles) else None

    def sample(xy) -> np.ndarray:
        xy = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
        result = np.full(len(xy), -np.inf)
        if tree is None or not len(xy):
            return np.full(len(xy), np.nan)
        pidx, tidx = tree.query(shapely.points(xy), predicate="intersects")
        tri = vertices[triangles[tidx]]
        p = xy[pidx]
        v0, v1, v2 = tri[:, 0, :2], tri[:, 1, :2], tri[:, 2, :2]
        den = (v1[:, 1] - v2[:, 1]) * (v0[:, 0] - v2[:, 0]) + (v2[:, 0] - v1[:, 0]) * (v0[:, 1] - v2[:, 1])
        ok = np.abs(den) > 1e-18
        den = np.where(ok, den, 1.0)
        w0 = ((v1[:, 1] - v2[:, 1]) * (p[:, 0] - v2[:, 0]) + (v2[:, 0] - v1[:, 0]) * (p[:, 1] - v2[:, 1])) / den
        w1 = ((v2[:, 1] - v0[:, 1]) * (p[:, 0] - v2[:, 0]) + (v0[:, 0] - v2[:, 0]) * (p[:, 1] - v2[:, 1])) / den
        z = w0 * tri[:, 0, 2] + w1 * tri[:, 1, 2] + (1.0 - w0 - w1) * tri[:, 2, 2]
        np.maximum.at(result, pidx[ok], z[ok])
        result[~np.isfinite(result)] = np.nan
        return result

    return sample


def _manhole_rings(manholes, segments: int, below: float) -> list[tuple[np.ndarray, tuple[float, float], float]]:
    """``(ring_xyz, centre, ring_radius)`` per manhole; the ring is circumscribed around the cover circle."""
    rings = []
    segments = max(int(segments), 6)
    for item in manholes or ():
        x, y, z, diameter = (float(v) for v in item[:4])
        if not (np.isfinite(x) and np.isfinite(y) and np.isfinite(z)) or diameter <= 0:
            continue
        radius = 0.5 * diameter / math.cos(math.pi / segments)
        angle = np.linspace(0.0, 2.0 * math.pi, segments, endpoint=False)
        ring = np.column_stack((x + radius * np.cos(angle), y + radius * np.sin(angle), np.full(segments, z - below)))
        rings.append((np.vstack((ring, ring[:1])), (x, y), radius))
    return rings


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


__all__ = ["select_tin_support", "build_labeled_tin", "tin_sampler", "missing_constraint_edges", "tin_quality", "label_boundary_consistent"]
