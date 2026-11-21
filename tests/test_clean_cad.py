"""Regression tests for clean CAD output from the outdoor chain."""
from __future__ import annotations

import numpy as np
import sys
import types
from shapely.geometry import MultiPolygon, Polygon, box

from samcloud.drone.contracts import LabeledTin, LocalFrame, PavementPolygon, PointSet, TIN_LABELS
from samcloud.drone.run_environment import _dump, _step_export
from samcloud.outdoor_model import pavement, terrain, vegetation
from samcloud.outdoor_model.scene import assemble_dxf, assemble_scene
from samcloud.terrain.tin import build_labeled_tin, select_tin_support


FRAME = LocalFrame((2_600_000.0, 1_200_000.0, 500.0))


def _road_tin(parts: list[tuple[float, float, float, float]]) -> LabeledTin:
    vertices: list[list[float]] = []
    triangles: list[list[int]] = []
    for xmin, ymin, xmax, ymax in parts:
        start = len(vertices)
        vertices.extend([[xmin, ymin, 0.0], [xmax, ymin, 0.0], [xmax, ymax, 0.0], [xmin, ymax, 0.0]])
        triangles.extend([[start, start + 1, start + 2], [start, start + 2, start + 3]])
    return LabeledTin(np.asarray(vertices), np.asarray(triangles), np.full(len(triangles), TIN_LABELS["road"], np.uint8))


def _pavement_config(**outline: float) -> dict:
    return {"pavement": {"min_thickness_m": 0.1, "min_area_m2": 0.01,
            "outline": {"close_m": 0.0, "open_m": 0.0, "min_area_m2": 0.01, "min_hole_m2": 1.0,
                        "simplify_m": 0.0, **outline}}, "dxf": {}}


def test_cleaned_pavement_is_the_tin_constraint() -> None:
    main = box(0, 0, 6, 4).difference(box(2.9, 3.8, 3.1, 4.2))
    source = MultiPolygon([main, box(9, 0, 9.3, 0.3)])
    cfg = _pavement_config(close_m=0.25, open_m=0.1, min_area_m2=1.0)
    topology = pavement.pavement_topology([PavementPolygon(1, "road", source)], config=cfg)
    assert len(topology.polygons) == 1
    cleaned = topology.polygons[0].polygon
    assert cleaned.area > main.area
    grid = np.array([[x, y, 0.01 * x] for x in np.arange(-1.0, 8.0, 0.5) for y in np.arange(-1.0, 6.0, 0.5)])
    tin = build_labeled_tin(grid, regions=topology.regions(), max_steiner_per_edge=4)
    centres = tin.vertices[tin.triangles].mean(axis=1)
    terrain_inside = (tin.labels == TIN_LABELS["terrain"]) & np.asarray(
        __import__("shapely").contains_xy(cleaned, centres[:, 0], centres[:, 1]), dtype=bool,
    )
    assert not terrain_inside.any()


def test_disjoint_pavement_footprint_becomes_stable_elements() -> None:
    tin = _road_tin([(0, 0, 2, 2), (5, 0, 7, 2)])
    elements, _ = pavement.build_pavement_elements([PavementPolygon(2, "road", box(-1, -1, 8, 3))], [], [], tin,
                                                    config=_pavement_config())
    dxf = pavement.to_dxf_entities(elements, FRAME, config=_pavement_config())
    assert [element.id for element in elements] == ["pavement-road-2_1", "pavement-road-2_2"]
    assert len(dxf["S-PAVE"]) == 2


def test_pavement_tin_surface_is_watertight_with_hole() -> None:
    vertices = np.array([
        [0, 0, 1], [4, 0, 1], [4, 4, 1], [0, 4, 1],
        [1, 1, 1], [3, 1, 1], [3, 3, 1], [1, 3, 1],
    ], dtype=float)
    triangles = np.array([
        [0, 1, 5], [0, 5, 4], [1, 2, 6], [1, 6, 5],
        [2, 3, 7], [2, 7, 6], [3, 0, 4], [3, 4, 7],
    ])
    tin = LabeledTin(vertices, triangles, np.full(len(triangles), TIN_LABELS["road"], np.uint8))
    footprint = box(0, 0, 4, 4).difference(box(1, 1, 3, 3))
    elements, qa = pavement.build_pavement_elements(
        [PavementPolygon(3, "road", footprint)], [], [], tin, config=_pavement_config(), topology_ready=True,
    )
    assert not qa and len(elements) == 1
    from samcloud.bim.core.ifc.outdoor import _pavement_mesh
    _vertices, faces = _pavement_mesh(elements[0])
    edges: dict[tuple[int, int], int] = {}
    for face in faces:
        for first, second in zip(face, face[1:] + face[:1]):
            edge = tuple(sorted((first, second)))
            edges[edge] = edges.get(edge, 0) + 1
    assert edges and set(edges.values()) == {2}


def test_terrain_contours_and_spots_stay_in_measured_cells() -> None:
    rows, cols = np.indices((12, 12))
    dtm = (rows + cols).astype(float) * 0.2
    validity = np.zeros_like(dtm, dtype=np.uint8)
    validity[:, :6] = 1
    vertices = np.array([[0, 0, 0], [12, 0, 2.4], [12, 12, 4.8], [0, 12, 2.4]], float)
    tin = LabeledTin(vertices, np.array([[0, 1, 2], [0, 2, 3]]), np.zeros(2, np.uint8))
    entities, _ = terrain.terrain_dxf_entities([], FRAME, tin=tin, dtm=dtm, validity=validity,
                                                dtm_bounds=(0, 0, 12, 12), resolution=1.0,
                                                config={"dxf": {"contour_major_m": 1.0, "contour_minor_m": 0.5,
                                                                "spline_contours": True, "spot_spacing_m": 2.0,
                                                                "spot_heights": True}})
    assert entities.get("S-TOPO-MAJR") or entities.get("S-TOPO-MINR")
    assert entities.get("S-ELEV")


def test_lawn_areas_export_closed_vegetation_polylines() -> None:
    areas = vegetation.build_lawn_areas([box(0, 0, 3, 2)], lambda xy: np.zeros(len(xy)))
    entities = vegetation.to_dxf_entities([], FRAME, lawn_areas=areas)
    assert len(entities["S-VEG"]) == 1
    assert entities["S-VEG"][0]["closed"] is True


def test_empty_clipped_lawn_is_absent_from_scene_and_dxf() -> None:
    areas = vegetation.build_lawn_areas([Polygon()], lambda xy: np.zeros(len(xy)))
    elements = vegetation.lawn_elements(areas)
    entities = vegetation.to_dxf_entities([], FRAME, lawn_areas=areas)
    assert not areas and not elements and "S-VEG" not in entities
    assert vegetation.category_counts(elements, areas)["vegetation_area"] == 0


def test_tin_support_reduces_noisy_surface() -> None:
    rng = np.random.default_rng(33)
    xy = rng.uniform(0, 40, (20_000, 2))
    z = 0.01 * xy[:, 0] - 0.005 * xy[:, 1]
    z[rng.choice(len(z), len(z) // 100, replace=False)] += 3.0
    selected, report = select_tin_support(np.column_stack((xy, z)), tolerance=0.03, initial_spacing=2.0, max_vertices=5_000)
    assert len(selected) < len(xy) * 0.1
    assert report["vertices"] == len(selected)


def test_export_publishes_and_ignores_publish_failure(tmp_path, monkeypatch) -> None:
    project = tmp_path / "project"
    output = project / "products" / "environment"
    cache = output / "cache"
    cache.mkdir(parents=True)
    pointset = PointSet(np.array([[0.0, 0.0, 0.0]]), np.array([2], np.uint16))
    _dump(cache / "load.pkl", (pointset, FRAME))
    _dump(cache / "scene.pkl", {"scene": assemble_scene(frame=FRAME), "dxf": assemble_dxf({}, FRAME)})
    module = types.ModuleType("samcloud.drone.publish")
    module.publish_environment_exports = lambda _project, exports: {"dxf": str(exports["dxf"])}
    monkeypatch.setitem(sys.modules, "samcloud.drone.publish", module)
    monkeypatch.setattr("samcloud.drone.run_environment._dsm_mesh", lambda *_args: None)
    result = _step_export(project, output, {"dxf": {}}, cache)
    assert result["published"] == {"dxf": result["dxf"]}
    module.publish_environment_exports = lambda *_args: (_ for _ in ()).throw(RuntimeError("offline"))
    failed = _step_export(project, output, {"dxf": {}}, cache)
    assert failed["publish_error"] == "offline"
