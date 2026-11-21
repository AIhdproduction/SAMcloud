"""Regression tests for error-driven terrain TIN support selection."""
from __future__ import annotations

import time

import numpy as np
from shapely.geometry import Polygon, box

from samcloud.drone.contracts import TIN_LABELS
from samcloud.terrain.tin import _interpolate_z, _unique_vertices, build_labeled_tin, missing_constraint_edges, select_tin_support


def _old_unique_vertices(ground: np.ndarray, segments, sampler):
    """Pre-vectorisation implementation retained as an exact behaviour oracle."""
    records = []
    lookup = {}
    scale = 1e9

    def add(point, prefer=False):
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


def test_planar_ground_is_reduced_below_two_percent_with_bounded_error():
    rng = np.random.default_rng(5)
    xy = rng.uniform(0.0, 60.0, (200_000, 2))
    ground = np.column_stack((xy, 0.012 * xy[:, 0] - 0.008 * xy[:, 1] + rng.normal(0.0, 0.005, len(xy))))
    selected, report = select_tin_support(ground, tolerance=0.03, initial_spacing=2.0, max_vertices=200_000)
    assert len(selected) < len(ground) * 0.02
    assert report["max_error_m"] <= 0.035


def test_support_points_concentrate_at_embankment_and_curb_breaks():
    rng = np.random.default_rng(7)
    xy = rng.uniform(0.0, 60.0, (180_000, 2))
    embankment = np.clip((xy[:, 0] - 29.5) / 0.5, 0.0, 1.0) * 0.5
    curb = np.where(xy[:, 0] >= 45.0, 0.12, 0.0)
    ground = np.column_stack((xy, 0.003 * xy[:, 1] + embankment + curb + rng.normal(0.0, 0.002, len(xy))))
    selected, report = select_tin_support(ground, tolerance=0.03, initial_spacing=2.0, max_vertices=20_000)
    near_break = ((np.abs(selected[:, 0] - 30.0) < 1.5) | (np.abs(selected[:, 0] - 45.0) < 1.5)).sum() / (6.0 * 60.0)
    flat = ((selected[:, 0] < 20.0) | ((selected[:, 0] > 34.0) & (selected[:, 0] < 40.0)) | (selected[:, 0] > 50.0)).sum() / (46.0 * 60.0)
    assert near_break >= 5.0 * flat
    assert report["p95_error_m"] <= 0.03


def test_support_selection_honours_vertex_cap():
    rng = np.random.default_rng(11)
    xy = rng.uniform(0.0, 40.0, (80_000, 2))
    ground = np.column_stack((xy, np.sin(xy[:, 0]) * np.cos(xy[:, 1]) * 0.2))
    selected, report = select_tin_support(ground, tolerance=0.002, initial_spacing=1.0, max_vertices=300)
    assert len(selected) <= 300
    assert report["vertices"] == len(selected)


def test_vectorised_unique_vertices_matches_previous_breakline_precedence():
    rng = np.random.default_rng(17)
    ground = np.column_stack((rng.integers(0, 15, (300, 2)), rng.normal(size=300))).astype(float)
    ground[::9, :2] = ground[0, :2]
    segments = [
        (np.array([1.0, 2.0, np.nan]), np.array([4.0, 2.0, 1.0]), 0),
        (np.array([1.0, 2.0, 3.0]), np.array([7.0, 8.0, np.nan]), 1),
        (np.array([7.0, 8.0, 5.0]), np.array([1.0, 2.0, 4.0]), 2),
    ]
    expected_vertices, expected_indexed = _old_unique_vertices(ground, segments, None)
    vertices, indexed = _unique_vertices(ground, segments, None)
    assert np.array_equal(vertices, expected_vertices)
    assert indexed == expected_indexed


def test_simplified_labeled_tin_recovers_constraints_under_twenty_seconds():
    x, y = np.meshgrid(np.linspace(0.0, 60.0, 600), np.linspace(0.0, 50.0, 500))
    xy = np.column_stack((x.ravel(), y.ravel()))
    ground = np.column_stack((xy, 0.01 * xy[:, 0] - 0.006 * xy[:, 1]))
    regions = {
        "road": [box(5.0, 5.0, 40.0, 18.0)],
        "sidewalk": [Polygon([(8.0, 28.0), (48.0, 28.0), (48.0, 36.0), (8.0, 36.0)])],
        "path": [],
        "lawn": [],
    }
    curb = np.array([[4.0, 22.0, -0.05], [56.0, 22.0, 0.47]])
    started = time.perf_counter()
    tin = build_labeled_tin(
        ground, regions=regions, breaklines=[curb], tolerance=0.03, initial_spacing=2.0, max_vertices=200_000,
    )
    elapsed = time.perf_counter() - started
    assert elapsed < 20.0, f"simplified 300k TIN took {elapsed:.2f} s"
    assert missing_constraint_edges(tin) == []
    assert set(tin.labels) & {TIN_LABELS["road"], TIN_LABELS["sidewalk"]}


def test_dtm_raster_refinement_keeps_curb_edges_and_bounded_error():
    """A 300-square DTM is simplified without losing its constrained curb."""
    axis = np.arange(300, dtype=float) * 0.25
    x, y = np.meshgrid(axis, axis)
    rng = np.random.default_rng(41)
    curb_x = 37.5
    z = 0.008 * x - 0.004 * y + np.where(x >= curb_x, 0.15, 0.0)
    ground = np.column_stack((x.ravel(), y.ravel(), (z + rng.normal(0.0, 0.01, z.shape)).ravel()))
    curb = np.array([[curb_x, 0.0, np.nan], [curb_x, axis[-1], np.nan]])
    started = time.perf_counter()
    tin = build_labeled_tin(
        ground, regions={"road": [], "sidewalk": [], "path": [], "lawn": []}, breaklines=[curb],
        tolerance=0.03, initial_spacing=2.0, max_vertices=20_000,
    )
    elapsed = time.perf_counter() - started
    assert elapsed < 10.0, f"raster TIN took {elapsed:.2f} s"
    assert tin.simplify_report["max_error_m"] <= 0.08
    assert missing_constraint_edges(tin) == []


def _interpolate_with_tin(triangulation, z: np.ndarray, xy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Use the public mesh representation to check its candidate error."""
    simplex = triangulation.find_simplex(xy)
    values = np.full(len(xy), np.nan)
    inside = simplex >= 0
    transform = triangulation.transform[simplex[inside], :2]
    weights = np.einsum("nij,nj->ni", transform, xy[inside] - triangulation.transform[simplex[inside], 2])
    barycentric = np.column_stack((weights, 1.0 - weights.sum(axis=1)))
    values[inside] = np.einsum("ni,ni->n", barycentric, z[triangulation.simplices[simplex[inside]]])
    return values, simplex


def test_refinement_reports_one_candidate_per_triangle_per_round():
    rng = np.random.default_rng(43)
    xy = rng.uniform(0.0, 30.0, (12_000, 2))
    ground = np.column_stack((xy, 0.03 * np.sin(xy[:, 0]) * np.cos(xy[:, 1])))
    _selected, report = select_tin_support(ground, tolerance=0.003, initial_spacing=2.0, max_vertices=4_000)
    assert report["max_added_per_triangle_per_round"] == 1
    assert all(added <= triangles for added, triangles in zip(report["round_additions"], report["round_triangle_counts"]))
