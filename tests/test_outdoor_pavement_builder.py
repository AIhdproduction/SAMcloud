"""Tests for samcloud.outdoor_model.pavement (synthetic road/sidewalk/path scene with noise)."""

from __future__ import annotations

import ezdxf
import ifcopenshell
import numpy as np
import pytest
from shapely.geometry import Polygon, box

from samcloud.bim.core.ifc import write_ifc
from samcloud.bim.core.model import PavementElement, Scene
from samcloud.drone import contracts as c
from samcloud.drone import qa_cad
from samcloud.dxf.las_source import Frame
from samcloud.dxf.result import BackgroundImage, DxfPlanResult
from samcloud.dxf.writer import build_dxf_document
from samcloud.environment import topology
from samcloud.outdoor_model import pavement as pv

pytest.importorskip("samcloud.terrain.tin")
from samcloud.terrain.tin import build_labeled_tin  # noqa: E402

E0, N0, H0 = 2_600_000.0, 1_200_000.0, 430.0
FRAME = c.LocalFrame(origin_lv95=(E0, N0, H0))
CFG = c.load_drone_config()


def _scene(slope: float = 0.02, seed: int = 3):
    """Road (gap + overlap at the curb), sidewalk, path inside a lawn, curb axis at y = 10."""
    road = Polygon([(0, 0), (30, 0), (30, 9.93), (15, 9.93), (15, 10.07), (0, 10.07)])
    sidewalk = box(0, 10.0, 30, 14)
    path = box(5, 16, 25, 19)
    lawn = box(0, 14, 30, 20)
    polys = [c.PavementPolygon(0, "road", road), c.PavementPolygon(1, "sidewalk", sidewalk),
             c.PavementPolygon(2, "path", path)]
    curb = c.CurbAxis(0, np.array([[0.0, 10.0, 0.10], [30.0, 10.0, 0.10 + 30 * slope]]))
    rng = np.random.default_rng(seed)
    gx, gy = np.meshgrid(np.arange(-1, 31.5, 0.9), np.arange(-1, 21.5, 0.9))
    xy = np.column_stack([gx.ravel(), gy.ravel()]) + rng.normal(0, 0.1, (gx.size, 2))
    z = slope * xy[:, 0] + 0.01 * slope * xy[:, 1] + rng.normal(0, 0.004, len(xy))
    return polys, [lawn], [curb], np.column_stack([xy, z])


def _tin(polys, lawns, curbs, ground):
    topo = pv.pavement_topology(polys, lawns, curbs, config=CFG)
    return topo, build_labeled_tin(ground, regions=topo.regions(), breaklines=[curbs[0].points])


@pytest.fixture(scope="module")
def built():
    polys, lawns, curbs, ground = _scene()
    topo, tin = _tin(polys, lawns, curbs, ground)
    elements, qa = pv.build_pavement_elements(polys, lawns, curbs, tin, config=CFG)
    return polys, lawns, curbs, topo, tin, elements, qa


def _write_dxf(layers: dict, path) -> None:
    result = DxfPlanResult("t", "site", "outdoor", None, 1.0, Frame(np.zeros(3)), "t", layers, {},
                           dict.fromkeys(layers, 7), BackgroundImage(None, None, (E0, N0, E0 + 10, N0 + 10)))
    build_dxf_document(result).saveas(path)


def _rules(results, prefix):
    found = [r for r in results if r["rule"].startswith(prefix)]
    assert found, prefix
    return found


def test_topology_closes_gap_and_overlap_below_1cm2(built) -> None:
    polys, lawns, curbs, topo, _tin_, elements, _qa = built
    raw = {k: [p.polygon for p in polys if p.kind == k] for k in c.PAVEMENT_KINDS}
    before = qa_cad.check_topology(raw, CFG)
    assert not all(r["ok"] for r in before), "the synthetic input must contain a gap/overlap"
    resolved = {k: [p.polygon for p in topo.polygons if p.kind == k] for k in c.PAVEMENT_KINDS}
    after = qa_cad.check_topology(resolved, CFG)
    assert all(r["ok"] for r in after), after
    assert all(p.polygon.is_valid for p in topo.polygons)
    assert topo.worst_cm2() < CFG["pavement"]["max_gap_overlap_cm2"]
    # the built products (union of the TIN triangles) stay below the limit as well
    footprints = pv.element_footprints(elements)
    final = qa_cad.check_topology(footprints, CFG)
    assert all(r["ok"] for r in final), final
    report = topology.pairwise_report({**resolved, "lawn": topo.lawns})
    assert all(r["ok"] for r in report.values())
    assert ("road", 0) in topo.snapped_ids and ("sidewalk", 1) not in topo.snapped_ids


def test_snap_search_radius_is_configurable() -> None:
    polys, lawns, curbs, _ = _scene()
    cfg = {**CFG, "pavement": {**CFG["pavement"], "snap_search_m": 0.01}}
    topo = pv.pavement_topology(polys, lawns, curbs, config=cfg)
    default = pv.pavement_topology(polys, lawns, curbs, config=CFG)
    assert topo.worst_cm2() >= default.worst_cm2()


def _edges(tris: np.ndarray) -> set[tuple[int, int]]:
    return {tuple(sorted(e)) for t in tris for e in ((t[0], t[1]), (t[1], t[2]), (t[2], t[0]))}


def test_pavement_top_vertices_are_shared_with_the_terrain_tin(built) -> None:
    _p, _l, _c, _topo, tin, elements, _qa = built
    assert {e.params.kind for e in elements} == {"road", "sidewalk", "path"}
    key = lambda v: tuple(np.round(np.asarray(v, dtype=float), 9))  # noqa: E731
    tin_vertices = {key(v) for v in tin.vertices}
    tin_edges = {tuple(sorted((key(tin.vertices[a]), key(tin.vertices[b])))) for a, b in
                 _edges(np.asarray(tin.triangles))}
    from samcloud.outdoor_model import terrain as tr
    terrain = tr.build_terrain_elements(tin, config=CFG)[0][0]
    terrain_vertices = {key(v) for v in terrain.params.vertices}
    shared = 0
    for e in elements:
        v = np.asarray(e.params.vertices)
        assert {key(p) for p in v} <= tin_vertices
        boundary = [edge for edge, n in _count_edges(e.params.triangles).items() if n == 1]
        assert boundary
        for a, b in boundary:
            assert tuple(sorted((key(v[a]), key(v[b])))) in tin_edges
        shared += len({key(p) for p in v} & terrain_vertices)
    assert shared > 20, "pavement and terrain must share boundary vertices"
    # no gap: pavement + terrain triangles tile exactly the area of the TIN triangles of those labels
    def area(verts, tris) -> float:
        a, b, cc = (np.asarray(verts)[np.asarray(tris)][:, i, :2] for i in range(3))
        return float(0.5 * np.abs((b[:, 0] - a[:, 0]) * (cc[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (cc[:, 0] - a[:, 0])).sum())

    total = sum(area(e.params.vertices, e.params.triangles) for e in elements) + area(terrain.params.vertices, terrain.params.triangles)
    assert total == pytest.approx(area(tin.vertices, tin.triangles), rel=0.01)


def _count_edges(tris) -> dict[tuple[int, int], int]:
    counts: dict[tuple[int, int], int] = {}
    for a, b, cc in tris:
        for e in ((a, b), (b, cc), (cc, a)):
            counts[tuple(sorted(e))] = counts.get(tuple(sorted(e)), 0) + 1
    return counts


def test_pavement_thickness_and_closed_solid_in_ifc(built, tmp_path) -> None:
    *_, elements, _qa = built
    scene = Scene()
    for e in elements:
        scene.add(e)
    path = tmp_path / "pavement.ifc"
    write_ifc(scene, path)
    result = _rules(qa_cad.check_ifc(path, CFG), "9_")[0]
    assert result["ok"], result
    assert result["value"]["pavements"] == 3
    assert all(e.params.thickness >= CFG["pavement"]["min_thickness_m"] for e in elements)
    ifc = ifcopenshell.open(str(path))
    assert {p.ObjectType for p in ifc.by_type("IfcCivilElement")} == {"Strasse", "Trottoir", "Weg"}


def test_flat_pavement_still_has_min_thickness_in_ifc(tmp_path) -> None:
    polys, lawns, curbs, ground = _scene(slope=0.0005)
    ground[:, 2] = 0.0
    _topo, tin = _tin(polys, lawns, curbs, ground)
    elements, _qa = pv.build_pavement_elements(polys, lawns, curbs, tin, config=CFG)
    scene = Scene()
    for e in elements:
        scene.add(e)
    path = tmp_path / "flat.ifc"
    write_ifc(scene, path)
    assert _rules(qa_cad.check_ifc(path, CFG), "9_")[0]["ok"]


def test_dxf_outlines_pass_qa_rules_2_3_4_5(built, tmp_path) -> None:
    *_, elements, _qa = built
    entities = pv.to_dxf_entities(elements, FRAME, config=CFG)
    assert set(entities) == {"S-PAVE", "S-WALK", "S-PATH"}
    path = tmp_path / "pave.dxf"
    _write_dxf(entities, path)
    results = qa_cad.check_dxf(path, CFG)
    for prefix in ("2_", "3_", "4_", "5_", "6_"):
        assert all(r["ok"] for r in _rules(results, prefix)), (prefix, results)
    doc = ezdxf.readfile(str(path))
    kinds = {e.dxftype() for e in doc.modelspace().query("*[layer=='S-PAVE']")}
    assert kinds == {"POLYLINE"}  # slope 2 %: Z range > 0.3 m -> closed polyline3d
    assert not doc.modelspace().query("HATCH")


def test_dxf_flat_lwpolyline_at_median_z_and_hatch_only_on_request(tmp_path) -> None:
    polys, lawns, curbs, ground = _scene(slope=0.0005)
    _topo, tin = _tin(polys, lawns, curbs, ground)
    elements, _qa = pv.build_pavement_elements(polys, lawns, curbs, tin, config=CFG)
    entities = pv.to_dxf_entities(elements, FRAME, config=CFG)
    first = entities["S-PAVE"][0]
    assert first["type"] == "lwpolyline" and first["closed"] and not first["hatch"]
    assert first["elevation"] == pytest.approx(H0, abs=0.1)
    hatch_cfg = {**CFG, "dxf": {**CFG["dxf"], "hatch": True}}
    hatched = pv.to_dxf_entities(elements, FRAME, config=hatch_cfg)
    assert hatched["S-PAVE"][0]["hatch"] is True
    path = tmp_path / "flat.dxf"
    _write_dxf(hatched, path)
    assert all(r["ok"] for r in _rules(qa_cad.check_dxf(path, CFG), "2_"))
    assert ezdxf.readfile(str(path)).modelspace().query("HATCH")


def test_polygon_with_hole_gets_hole_polyline(built) -> None:
    *_, elements, _qa = built
    lawn_hole = [e for e in elements if e.params.kind == "sidewalk"]
    entities = pv.to_dxf_entities(lawn_hole, FRAME, config=CFG)
    assert len(entities["S-WALK"]) == 1  # sidewalk has no hole
    donut = box(0, 0, 10, 10).difference(box(4, 4, 6, 6))
    rng = np.random.default_rng(1)
    ground = np.column_stack([rng.uniform(-1, 11, (300, 2)), rng.normal(0, 0.003, 300)])
    tin = build_labeled_tin(ground, regions={"road": [donut]})
    elements, _ = pv.build_pavement_elements([c.PavementPolygon(0, "road", donut)], [], [], tin, config=CFG)
    assert len(elements) == 1
    rings = pv.to_dxf_entities(elements, FRAME, config=CFG)["S-PAVE"]
    assert len(rings) == 2 and all(r["closed"] for r in rings)


def test_missing_tin_triangles_and_rejected_polygons_become_qa_entries(built) -> None:
    polys, lawns, curbs, _t, tin, _e, _q = built
    far = c.PavementPolygon(7, "path", box(100, 100, 110, 110))
    bad = c.PavementPolygon(8, "road", box(50, 50, 60, 60), status="rejected", reason="test")
    elements, qa = pv.build_pavement_elements([*polys, far, bad], lawns, curbs, tin, config=CFG)
    assert len(elements) == 3
    reasons = {(q.category, q.element_id) for q in qa}
    assert ("pavement", "pavement-path-7") in reasons
    assert ("pavement", "road-8") in reasons


def test_category_counts_and_param_origins(built) -> None:
    *_, elements, _qa = built
    assert pv.category_counts(elements) == {"road": 1, "sidewalk": 1, "path": 1}
    road = next(e for e in elements if e.params.kind == "road")
    assert road.param_origin["thickness"].value == "assumed"
    assert road.param_origin["vertices"].value == "snapped"
    assert isinstance(road, PavementElement) and road.source_instance_ids == ["0"]
