"""Regression coverage for per-element outdoor pipeline failures."""
from __future__ import annotations

import pickle

import numpy as np
from shapely.geometry import Polygon

from samcloud.buildings.facades import facade_surfaces
from samcloud.drone.contracts import BuildingInstance, BuildingModel, CurbAxis, Evidence, LabeledTin, PavementPolygon, PointSet, TIN_LABELS, load_drone_config
from samcloud.drone.run_environment import _step_buildings
from samcloud.outdoor_model.pavement import build_pavement_elements, pavement_topology
from samcloud.terrain.tin import build_labeled_tin, missing_constraint_edges


def _wall_model() -> BuildingModel:
    vertices = np.array([
        [0.0, 0.0, 0.0], [4.0, 0.0, 0.0], [4.0, 0.0, 3.0], [0.0, 0.0, 3.0],
        [2.0, 0.0, 0.0],
    ])
    faces = np.array([[0, 1, 2], [0, 2, 3], [0, 4, 1]])
    return BuildingModel(7, "2.2", vertices, faces, np.ones(len(faces), dtype=np.uint8), "ok")


def test_facades_skip_and_report_degenerate_wall_triangles() -> None:
    report: dict = {}
    surfaces = facade_surfaces(_wall_model(), [], report=report)

    assert surfaces
    assert report["degenerate_wall_triangles"] == 1


def test_step_buildings_isolates_one_reconstruction_failure(tmp_path, monkeypatch) -> None:
    cache = tmp_path / "cache"
    cache.mkdir()
    points = PointSet(
        xyz=np.array([[0, 0, 0], [4, 0, 0], [4, 4, 0], [0, 4, 0], [2, 2, 3]], dtype=float),
        classes=np.full(5, 6, dtype=np.uint16),
    )
    with (cache / "load.pkl").open("wb") as stream:
        pickle.dump((points, None), stream)
    with (cache / "terrain.pkl").open("wb") as stream:
        pickle.dump({"dtm": np.zeros((2, 2)), "bounds": (0.0, 0.0, 4.0, 4.0), "resolution": 4.0}, stream)
    instances = [BuildingInstance(1, np.arange(5)), BuildingInstance(2, np.arange(5))]
    with (cache / "semantics.pkl").open("wb") as stream:
        pickle.dump({"instances": instances}, stream)

    def reconstruct(_points, building_id, **_kwargs):
        if building_id == 1:
            raise RuntimeError("forced reconstruction error")
        return _wall_model().__class__(2, "2.2", _wall_model().vertices, _wall_model().faces, _wall_model().face_roles, "ok")

    import samcloud.buildings.reconstruct as reconstruct_module
    import samcloud.buildings.openings as openings_module

    monkeypatch.setattr(reconstruct_module, "reconstruct_building", reconstruct)
    monkeypatch.setattr(openings_module, "detect_building_openings", lambda *_args, **_kwargs: ([], []))
    result = _step_buildings(tmp_path, tmp_path, load_drone_config(), cache)
    stored = pickle.load((cache / "buildings.pkl").open("rb"))

    assert result["buildings"] >= 1
    assert any(item.element_id == "building-1" and "forced reconstruction error" in item.reason for item in stored["qa"])


def test_pavement_drops_none_curb_and_empty_polygon_with_qa() -> None:
    empty = PavementPolygon(5, "road", Polygon(), Evidence())
    tin = LabeledTin(np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
                     np.array([[0, 1, 2]]), np.array([TIN_LABELS["terrain"]], dtype=np.uint8))
    topology = pavement_topology([empty], [], [None], config=load_drone_config())
    _elements, qa = build_pavement_elements([empty], [], [None], tin, config=load_drone_config())

    assert len(topology.polygons) == 0
    assert {entry.element_id for entry in qa} >= {"road-5", "curb-0"}


def _components(tin, label: str) -> int:
    selected = np.flatnonzero(tin.labels == TIN_LABELS[label])
    neighbours: dict[int, set[int]] = {}
    for triangle in tin.triangles[selected]:
        for first, second in ((triangle[0], triangle[1]), (triangle[1], triangle[2]), (triangle[2], triangle[0])):
            neighbours.setdefault(int(first), set()).add(int(second))
            neighbours.setdefault(int(second), set()).add(int(first))
    seen: set[int] = set()
    count = 0
    for root in neighbours:
        if root in seen:
            continue
        count += 1
        stack = [root]
        seen.add(root)
        while stack:
            current = stack.pop()
            for other in neighbours[current]:
                if other not in seen:
                    seen.add(other)
                    stack.append(other)
    return count


def test_simplified_tin_keeps_nearby_curb_constraints_and_contiguous_labels() -> None:
    x, y = np.meshgrid(np.linspace(-1.0, 11.0, 121), np.linspace(-1.0, 5.0, 61))
    ground = np.column_stack((x.ravel(), y.ravel(), 0.01 * x.ravel() - 0.002 * y.ravel()))
    regions = {
        "road": [Polygon([(0, 0), (10, 0), (10, 2), (0, 2)])],
        "sidewalk": [Polygon([(0, 2), (10, 2), (10, 4), (0, 4)])],
        "path": [], "lawn": [],
    }
    curb = np.array([[0.0, 2.03, 0.0], [10.0, 2.03, 0.1]])
    tin = build_labeled_tin(ground, regions=regions, breaklines=[curb], tolerance=0.03, initial_spacing=2.0, max_vertices=5_000)

    assert missing_constraint_edges(tin) == []
    assert _components(tin, "road") == 1
    assert _components(tin, "sidewalk") == 1
    assert len(tin.vertices) < len(ground) * 0.5
