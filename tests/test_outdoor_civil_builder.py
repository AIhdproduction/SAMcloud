"""Tests for samcloud.outdoor_model.civil (curbs, markings, manholes; synthetic noisy inputs)."""

from __future__ import annotations

import math

import ezdxf
import ifcopenshell
import numpy as np
import pytest

from samcloud.bim.core.ifc import write_ifc
from samcloud.bim.core.model import CurbElement, ManholeElement, MarkingElement, Scene
from samcloud.drone import contracts as c
from samcloud.drone import qa_cad
from samcloud.dxf.las_source import Frame
from samcloud.dxf.result import BackgroundImage, DxfPlanResult
from samcloud.dxf.writer import build_dxf_document
from samcloud.environment.curbs import split_straight_and_arcs
from samcloud.outdoor_model import civil

E0, N0, H0 = 2_600_000.0, 1_200_000.0, 430.0
FRAME = c.LocalFrame(origin_lv95=(E0, N0, H0))
CFG = c.load_drone_config()


def _straight_plus_arc() -> np.ndarray:
    rng = np.random.default_rng(5)
    straight = [(0.0, 0.0, 0.10), (6.0, 0.0, 0.11), (12.0, 0.0, 0.12)]
    arc = [(12 + 8 * math.cos(math.radians(-90 + 11 * k)), 8 + 8 * math.sin(math.radians(-90 + 11 * k)), 0.12 + 0.002 * k)
           for k in range(1, 9)]
    pts = np.array(straight + arc)
    pts[1:, :2] += rng.normal(0, 0.0015, (len(pts) - 1, 2))
    return pts


def _full_arc() -> np.ndarray:
    return np.array([(20 + 8 * math.cos(0.1 * i), 8 * math.sin(0.1 * i), 0.003 * (-1) ** i) for i in range(10)])


def _disc(rng, centre, diameter, n, z):
    r = diameter / 2 * np.sqrt(rng.uniform(0, 1, n))
    a = rng.uniform(0, 2 * np.pi, n)
    return np.column_stack([centre[0] + r * np.cos(a), centre[1] + r * np.sin(a), z + rng.normal(0, 0.003, n)])


def _pointset(parts: list[tuple[np.ndarray, int]]) -> c.PointSet:
    xyz = np.vstack([p for p, _ in parts])
    classes = np.concatenate([np.full(len(p), code, dtype=np.uint16) for p, code in parts])
    return c.PointSet(xyz=xyz, classes=classes)


def _write_dxf(layers: dict, path) -> None:
    result = DxfPlanResult("t", "site", "outdoor", None, 1.0, Frame(np.zeros(3)), "t", layers, {},
                           dict.fromkeys(layers, 7), BackgroundImage(None, None, (E0, N0, E0 + 10, N0 + 10)))
    build_dxf_document(result).saveas(path)


@pytest.fixture(scope="module")
def built():
    rng = np.random.default_rng(11)
    curbs = [c.CurbAxis(0, _straight_plus_arc(), evidence=c.Evidence(points=300)),
             c.CurbAxis(1, _full_arc(), evidence=c.Evidence(points=200))]
    ps = _pointset([
        (_disc(rng, (3.0, 4.0), 0.62, 400, 1.25), c.MANHOLE),
        (np.column_stack([rng.uniform(5, 7, 600), rng.normal(6, 0.05, 600), 1.2 + rng.normal(0, 0.003, 600)]), c.MANHOLE),
        (_disc(rng, (9.0, 9.0), 0.5, 10, 1.2), c.MANHOLE),
        (np.column_stack([rng.uniform(0, 3.0, 800), rng.uniform(0, 0.12, 800) + 3.0, 1.1 + rng.normal(0, 0.003, 800)]), c.ROAD_MARKING),
        (np.column_stack([rng.uniform(6, 6.8, 1000), rng.uniform(1, 1.5, 1000), 1.1 + rng.normal(0, 0.003, 1000)]), c.ROAD_MARKING),
        (_disc(rng, (9.0, 1.0), 0.1, 8, 1.1), c.ROAD_MARKING),
    ])
    markings = civil.marking_features(ps)
    manholes = civil.manhole_detections(ps)
    elements, qa = civil.build_civil_elements(curbs, markings, manholes, config=CFG)
    return curbs, ps, markings, manholes, elements, qa


def test_manhole_detection_diameter_centre_and_cover_height(built) -> None:
    _c, _ps, _m, manholes, elements, qa = built
    ok = [m for m in manholes if m.status == "ok"]
    assert len(ok) == 1
    assert ok[0].diameter == pytest.approx(0.62, abs=0.05)
    assert ok[0].center[:2] == pytest.approx([3.0, 4.0], abs=0.05)
    assert ok[0].center[2] == pytest.approx(1.25, abs=0.005)  # Z_DECKEL = median Z
    rejected = [m for m in manholes if m.status == "rejected"]
    assert len(rejected) == 1 and "round" in rejected[0].reason
    assert any(q.category == "manhole" for q in qa)


def test_marking_features_line_polygon_and_reject(built) -> None:
    _c, _ps, markings, _m, elements, qa = built
    lines = [m for m in markings if m.kind == "line" and m.status == "ok"]
    polys = [m for m in markings if m.kind == "polygon" and m.status == "ok"]
    assert len(lines) == 1 and len(polys) == 1
    assert lines[0].width == pytest.approx(0.12, abs=0.03)
    assert np.ptp(lines[0].points[:, 0]) == pytest.approx(3.0, abs=0.25)
    assert polys[0].closed and len(polys[0].points) >= 4
    from shapely.geometry import Polygon
    assert Polygon(polys[0].points[:, :2]).area == pytest.approx(0.8 * 0.5, rel=0.15)
    assert any(m.status == "rejected" for m in markings)
    assert any(q.category == "marking" for q in qa)


def test_curb_straight_plus_arc_passes_ifc_rule_8(built, tmp_path) -> None:
    curbs, _ps, _m, _mh, elements, _qa = built
    parts = split_straight_and_arcs(curbs[0].points)
    assert [k for k, _ in parts] == ["straight", "arc"]
    curb_elements = [e for e in elements if isinstance(e, CurbElement)]
    assert len(curb_elements) == 2
    mixed, arc = curb_elements
    assert mixed.evidence["straight_parts"] == 1 and mixed.evidence["arc_parts"] == 1
    assert mixed.params.width == CFG["curb"]["section_width_m"] and mixed.params.curb_sweep == "auto"
    scene = Scene()
    for e in curb_elements:
        scene.add(e)
    path = tmp_path / "curbs.ifc"
    write_ifc(scene, path)
    rule8 = [r for r in qa_cad.check_ifc(path, CFG) if r["rule"].startswith("8_")][0]
    assert rule8["ok"] and rule8["value"] == 2, rule8
    ifc = ifcopenshell.open(str(path))
    assert len(ifc.by_type("IfcFixedReferenceSweptAreaSolid")) == 1  # the continuous arc is swept
    assert ifc.by_type("IfcExtrudedAreaSolid")


def test_curb_extrude_only_config_is_passed_to_the_element(built) -> None:
    curbs = built[0]
    cfg = {**CFG, "ifc": {**CFG["ifc"], "curb_sweep": "extrude_only"}}
    elements, _qa = civil.build_civil_elements(curbs, config=cfg)
    assert {e.params.curb_sweep for e in elements} == {"extrude_only"}


def test_manhole_insert_with_z_deckel_typ_and_ifc_chamber(built, tmp_path) -> None:
    _c, _ps, _m, _mh, elements, _qa = built
    manhole = [e for e in elements if isinstance(e, ManholeElement)]
    assert len(manhole) == 1 and manhole[0].params.type == "Schacht"
    entities = civil.to_dxf_entities(manhole, FRAME, config=CFG)
    insert = entities["S-MANH"][0]
    assert insert["name"] == "MANHOLE" and insert["layer"] == "S-MANH"
    assert insert["attributes"] == {"Z_DECKEL": f"{H0 + manhole[0].params.center[2]:.2f}", "TYP": "Schacht"}
    assert insert["xscale"] == pytest.approx(manhole[0].params.diameter / 0.6)
    assert insert["point"][0] == pytest.approx(E0 + 3.0, abs=0.05)
    dxf = tmp_path / "m.dxf"
    _write_dxf(entities, dxf)
    loaded = ezdxf.readfile(str(dxf)).modelspace().query("INSERT")
    assert {a.dxf.tag: a.dxf.text for a in loaded[0].attribs} == {"Z_DECKEL": "431.25", "TYP": "Schacht"}
    scene = Scene()
    scene.add(manhole[0])
    ifc_path = tmp_path / "m.ifc"
    write_ifc(scene, ifc_path)
    chamber = ifcopenshell.open(str(ifc_path)).by_type("IfcDistributionChamberElement")
    assert len(chamber) == 1 and chamber[0].PredefinedType == "MANHOLE" and chamber[0].ObjectType == "Schacht"
    counts = qa_cad.count_consistency(civil.category_counts(manhole), dxf, ifc_path)
    assert all(r["ok"] for r in counts if r["rule"] == "15_count_manhole"), counts


def test_dxf_passes_qa_rules_1_5_6_and_marking_layer(built, tmp_path) -> None:
    _c, _ps, _m, _mh, elements, _qa = built
    entities = civil.to_dxf_entities(elements, FRAME, config=CFG)
    assert set(entities) == {"S-CURB", "S-MARK", "S-MANH"}
    path = tmp_path / "civil.dxf"
    _write_dxf(entities, path)
    results = qa_cad.check_dxf(path, CFG)
    for prefix in ("1_", "5_", "6_"):
        assert all(r["ok"] for r in results if r["rule"].startswith(prefix)), (prefix, results)
    doc = ezdxf.readfile(str(path))
    marks = doc.modelspace().query("*[layer=='S-MARK']")
    assert len(marks) == 2 and {m.dxftype() for m in marks} == {"POLYLINE"}
    assert sorted(m.is_closed for m in marks) == [False, True]
    assert len(doc.modelspace().query("*[layer=='S-CURB']")) == 2


def test_curb_dxf_removes_collinear_vertices_but_keeps_corners() -> None:
    pts = np.array([[0, 0, 0.0], [2, 0.004, 0.0], [4, -0.003, 0.1], [6, 0.0, 0.1], [6.0, 3.0, 0.1], [6.0, 6.0, 0.1]])
    thin = civil.thin_collinear_xy(pts, 0.015)
    assert thin.tolist() == [[0, 0, 0.0], [6, 0.0, 0.1], [6.0, 6.0, 0.1]]
    elements, _ = civil.build_civil_elements([c.CurbAxis(0, pts, evidence=c.Evidence(points=100))], config=CFG)
    line = civil.to_dxf_entities(elements, FRAME, config=CFG)["S-CURB"][0]
    assert line["type"] == "polyline3d" and len(line["points"]) == 3
    assert line["points"][0][0] == pytest.approx(E0)


def test_below_threshold_and_rejected_instances_become_qa_entries() -> None:
    short = c.CurbAxis(0, np.array([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]]))
    bad = c.CurbAxis(1, np.array([[0.0, 0.0, 0.0], [5.0, 0.0, 0.0]]), status="rejected", reason="test")
    good = c.CurbAxis(2, np.array([[0.0, 0.0, 0.0], [5.0, 0.0, 0.0]]), status="warning", reason="sparse")
    elements, qa = civil.build_civil_elements([short, bad, good], config=CFG)
    assert [e.id for e in elements] == ["curb-2"]
    assert elements[0].status.value == "warning"
    assert {q.element_id for q in qa} == {"curb-0", "curb-1"}


def test_category_counts(built) -> None:
    *_, elements, _qa = built
    assert civil.category_counts(elements) == {"curb": 2, "gutter": 0, "marking": 2, "manhole": 1, "fence": 0, "site_wall": 0}
    assert sum(isinstance(e, MarkingElement) for e in elements) == 2
    origins = next(e for e in elements if isinstance(e, ManholeElement)).param_origin
    assert origins["type"].value == "assumed" and origins["center"].value == "measured"
