from __future__ import annotations

import logging
import time

import numpy as np
import pytest
from shapely.geometry import LineString, Point, Polygon, box
from shapely.validation import make_valid

from samcloud.drone import contracts as c
from samcloud.drone.contracts import TIN_LABELS
from samcloud.outdoor_model import pavement as pv
from samcloud.terrain.tin import (
    build_labeled_tin,
    label_boundary_consistent,
    missing_constraint_edges,
    tin_quality,
)


def _ground(step: float = 0.25, extent: float = 20.0) -> np.ndarray:
    axis = np.arange(0.0, extent + step * 0.5, step)
    x, y = np.meshgrid(axis, axis)
    rng = np.random.default_rng(20261120)
    z = 0.03 * x - 0.017 * y + rng.normal(0.0, 0.02, x.shape)
    return np.column_stack((x.ravel(), y.ravel(), z.ravel()))


def _regions() -> dict[str, list[Polygon]]:
    road = Polygon([(2.0, 3.0), (18.0, 3.0), (18.0, 6.0), (2.0, 6.0)])
    sidewalk = Polygon([(2.0, 6.0), (18.0, 6.0), (18.0, 7.2), (2.0, 7.2)])
    curve = LineString([(3.0, 13.0), (7.0, 15.0), (12.0, 14.0), (17.0, 17.0)])
    path = curve.buffer(0.45, cap_style="round", join_style="round", quad_segs=2)
    return {"road": [road], "sidewalk": [sidewalk], "path": [path], "lawn": []}


def _outdoor_builder_scene(spacing: float) -> tuple[dict[str, list[Polygon]], list[np.ndarray], np.ndarray]:
    """The shared synthetic scene used by the terrain and pavement builders."""
    road = Polygon([(0, 0), (30, 0), (30, 9.93), (15, 9.93), (15, 10.07), (0, 10.07)])
    polygons = [c.PavementPolygon(0, "road", road), c.PavementPolygon(1, "sidewalk", box(0, 10.0, 30, 14)),
                c.PavementPolygon(2, "path", box(5, 16, 25, 19))]
    lawns = [box(0, 14, 30, 20)]
    curbs = [c.CurbAxis(0, np.array([[0.0, 10.0, 0.10], [30.0, 10.0, 0.70]]))]
    rng = np.random.default_rng(3)
    gx, gy = np.meshgrid(np.arange(-1, 31.5, spacing), np.arange(-1, 21.5, spacing))
    xy = np.column_stack([gx.ravel(), gy.ravel()]) + rng.normal(0, 0.1, (gx.size, 2))
    z = 0.02 * xy[:, 0] + 0.0002 * xy[:, 1] + rng.normal(0, 0.004, len(xy))
    topology = pv.pavement_topology(polygons, lawns, curbs, config=c.load_drone_config())
    return topology.regions(), curbs, np.column_stack([xy, z])


@pytest.fixture(scope="module")
def labelled_tin():
    regions = _regions()
    curb = np.array([[2.0, 6.0, 0.75], [10.0, 6.0, 0.99], [18.0, 6.0, 1.23]])
    return build_labeled_tin(_ground(), regions=regions, breaklines=[curb]), regions, curb


def _coordinate_edges(mesh) -> set[tuple[tuple[float, float], tuple[float, float]]]:
    edges = set()
    for tri in mesh.triangles:
        for a, b in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
            pa = tuple(np.round(mesh.vertices[a, :2], 9))
            pb = tuple(np.round(mesh.vertices[b, :2], 9))
            edges.add(tuple(sorted((pa, pb))))
    return edges


def test_every_densified_constraint_segment_is_an_edge(labelled_tin):
    tin, regions, curb = labelled_tin
    assert missing_constraint_edges(tin) == []
    assert missing_constraint_edges(tin, {"regions": regions, "breaklines": [curb]}) == []


@pytest.mark.parametrize(("spacing", "scene_name"), [(2.0, "terrain_builder_s13"), (0.9, "pavement_builder")])
def test_outdoor_builder_constraints_are_recovered_without_warning(caplog, spacing, scene_name):
    regions, curbs, ground = _outdoor_builder_scene(spacing)
    with caplog.at_level(logging.WARNING, logger="samcloud.terrain.tin"):
        tin = build_labeled_tin(ground, regions=regions, breaklines=[curbs[0].points])
    assert missing_constraint_edges(tin) == [], scene_name
    assert not [record for record in caplog.records if "could not recover" in record.getMessage()]


def test_crossing_constraints_are_noded_and_recovered():
    axis = np.arange(0.0, 10.01, 0.25)
    x, y = np.meshgrid(axis, axis)
    ground = np.column_stack((x.ravel(), y.ravel(), np.zeros(x.size)))
    tin = build_labeled_tin(
        ground,
        regions={"road": [box(1.0, 1.0, 9.0, 9.0)]},
        breaklines=[np.array([[0.0, 5.0, 0.0], [10.0, 5.0, 0.0]])],
    )
    endpoints = np.vstack(tin.constraint_segments)[:, :2]
    assert np.any(np.linalg.norm(endpoints - (1.0, 5.0), axis=1) < 1e-9)
    assert np.any(np.linalg.norm(endpoints - (9.0, 5.0), axis=1) < 1e-9)
    assert missing_constraint_edges(tin) == []


def test_two_millimetre_parallel_constraints_are_handled_without_degenerate_triangles():
    axis = np.arange(0.0, 10.01, 0.25)
    x, y = np.meshgrid(axis, axis)
    ground = np.column_stack((x.ravel(), y.ravel(), np.zeros(x.size)))
    tin = build_labeled_tin(
        ground,
        regions={},
        breaklines=[
            np.array([[0.0, 5.0, 0.0], [10.0, 5.0, 0.0]]),
            np.array([[0.0, 5.002, 0.0], [10.0, 5.002, 0.0]]),
        ],
    )
    assert missing_constraint_edges(tin) == []
    assert tin_quality(tin)["degenerate_count"] == 0


def test_label_areas_and_boundaries(labelled_tin):
    tin, regions, _ = labelled_tin
    quality = tin_quality(tin)
    for name in ("road", "sidewalk", "path"):
        expected = sum(poly.area for poly in regions[name])
        assert quality["label_areas"][name] == pytest.approx(expected, rel=0.005)
    assert label_boundary_consistent(tin, regions) < 0.001


def test_pavement_and_terrain_reassemble_one_mesh(labelled_tin):
    tin, _, _ = labelled_tin
    terrain = tin.without("road", "sidewalk", "path")
    pavement = tin.select("road", "sidewalk", "path")
    terrain_vertices = {tuple(np.round(v[:2], 9)) for v in terrain.vertices}
    pavement_vertices = {tuple(np.round(v[:2], 9)) for v in pavement.vertices}
    original_vertices = {tuple(np.round(v[:2], 9)) for v in tin.vertices}
    assert terrain_vertices | pavement_vertices == original_vertices
    assert terrain_vertices & pavement_vertices
    assert _coordinate_edges(terrain) & _coordinate_edges(pavement)


def test_breakline_z_is_honoured(labelled_tin):
    tin, _, curb = labelled_tin
    line = LineString(curb[:, :2])
    on_axis = np.array([line.distance(Point(v[:2])) < 1e-9 for v in tin.vertices])
    assert on_axis.any()
    for vertex in tin.vertices[on_axis]:
        distance = line.project(Point(vertex[:2]))
        expected = 0.75 + (1.23 - 0.75) * distance / line.length
        assert vertex[2] == pytest.approx(expected, abs=0.001)


def test_quality_has_no_degenerate_triangles():
    road = Polygon([(2.0, 3.0), (18.0, 3.0), (18.0, 6.0), (2.0, 6.0)])
    tin = build_labeled_tin(_ground(), regions={"road": [road], "sidewalk": [], "path": [], "lawn": []})
    quality = tin_quality(tin)
    assert quality["degenerate_count"] == 0
    assert quality["share_below_min_angle"] < 0.15


def test_polygon_hole_labels_lawn_island():
    road = Polygon([(2, 2), (18, 2), (18, 18), (2, 18)], holes=[[(7, 7), (13, 7), (13, 13), (7, 13)]])
    lawn = Polygon([(7, 7), (13, 7), (13, 13), (7, 13)])
    tin = build_labeled_tin(_ground(), regions={"road": [road], "sidewalk": [], "path": [], "lawn": [lawn]})
    centres = tin.vertices[tin.triangles, :2].mean(axis=1)
    island = (centres[:, 0] > 7.1) & (centres[:, 0] < 12.9) & (centres[:, 1] > 7.1) & (centres[:, 1] < 12.9)
    assert island.any()
    assert np.all(tin.labels[island] == TIN_LABELS["lawn"])


def test_performance_500k_dtm_points_and_two_km_constraints():
    x, y = np.meshgrid(np.arange(1000.0), np.arange(500.0))
    ground = np.column_stack((x.ravel(), y.ravel(), 0.002 * x.ravel() - 0.001 * y.ravel()))
    road = Polygon([(100, 150), (900, 150), (900, 350), (100, 350)])  # 2 km perimeter
    started = time.perf_counter()
    tin = build_labeled_tin(ground, regions={"road": [road], "sidewalk": [], "path": [], "lawn": []})
    elapsed = time.perf_counter() - started
    assert len(tin.triangles) > 0
    assert elapsed < 60.0, f"500k TIN took {elapsed:.2f} s"


def test_near_duplicate_and_validated_self_touching_polygon_do_not_crash():
    self_touching = make_valid(Polygon([(4, 4), (10, 10), (4, 10), (10, 4), (4, 4)]))
    breakline = np.array([[1.0, 1.0, 0.0], [1.000001, 1.000001, 0.0], [16.0, 16.0, 0.5]])
    tin = build_labeled_tin(
        _ground(), regions={"road": [self_touching], "sidewalk": [], "path": [], "lawn": []}, breaklines=[breakline]
    )
    assert len(tin.triangles) > 0
    assert tin_quality(tin)["degenerate_count"] == 0
