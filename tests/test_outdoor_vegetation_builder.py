"""Vegetation builder: trees, shrubs, lawn areas -> scene elements, DXF entities, IFC, QA."""

from __future__ import annotations

from pathlib import Path

import ezdxf
import numpy as np
import pytest
from shapely.geometry import Polygon

from samcloud.bim.core.ifc import write_ifc
from samcloud.bim.core.model import ParamOrigin, Scene, ShrubElement, TreeElement
from samcloud.drone import qa_cad
from samcloud.drone.contracts import Evidence, LocalFrame, ShrubInstance, TreeInstance
from samcloud.dxf.las_source import Frame
from samcloud.dxf.result import BackgroundImage, DxfPlanResult
from samcloud.dxf.writer import build_dxf_document
from samcloud.outdoor_model import vegetation as veg

FRAME = LocalFrame(origin_lv95=(2_683_000.0, 1_248_000.0, 430.0))


def _dtm(xy: np.ndarray) -> np.ndarray:
    return 1.0 + 0.013 * xy[:, 0] + 0.007 * xy[:, 1]


def _tree(tid: int, xy, height=8.4, crown=5.0, trunk=0.35, status="ok", points=400) -> TreeInstance:
    return TreeInstance(tid, xy, 1.03, height, crown, trunk, np.arange(points), Evidence(5, points, 0.021),
                        status, "" if status == "ok" else "too small")


def _shrub(sid: int = 0) -> ShrubInstance:
    outline = Polygon([(10.03, 10.02), (12.11, 10.07), (12.4, 11.93), (10.2, 12.21)])
    return ShrubInstance(sid, outline, 1.37, Evidence(4, 80, None), "ok", "")


def _lawn(extent: float = 6.0) -> Polygon:
    return Polygon([(0.03, 0.02), (extent + 0.11, 0.07), (extent + 0.3, extent * 0.7 + 0.13), (0.2, extent * 0.7 + 0.21)])


def _dxf(layers, tmp_path: Path):
    result = DxfPlanResult(
        "p", "site", "outdoor", None, 1.0, Frame(np.zeros(3)), "t", layers, {}, dict.fromkeys(layers, 3),
        BackgroundImage(None, None, (0.0, 0.0, 10.0, 10.0)),
    )
    path = tmp_path / "veg.dxf"
    build_dxf_document(result).saveas(path)
    return path, ezdxf.readfile(path)


def _attrs(insert) -> dict[str, str]:
    return {a.dxf.tag: a.dxf.text for a in insert.attribs}


def test_two_trees_elements_and_dxf_inserts(tmp_path: Path) -> None:
    trees = [_tree(1, (3.0, 4.0)), _tree(2, (7.5, 2.2), height=6.1, crown=3.4, trunk=0.22)]
    elements, qa = veg.build_vegetation_elements(trees, [])
    assert [type(e) for e in elements] == [TreeElement, TreeElement] and qa == []
    assert elements[0].param_origin["trunk_diameter"] is ParamOrigin.MEASURED
    entities = veg.to_dxf_entities(elements, FRAME)
    inserts = entities["S-TREE"]
    assert len(inserts) == 2
    assert inserts[0]["name"] == "TREE" and inserts[0]["point"] == pytest.approx([2_683_003.0, 1_248_004.0, 431.03])
    assert inserts[0]["xscale"] == pytest.approx(2.5) and inserts[1]["xscale"] == pytest.approx(1.7)
    _, doc = _dxf(entities, tmp_path)
    found = sorted((e for e in doc.modelspace() if e.dxftype() == "INSERT"), key=lambda e: e.dxf.insert.x)
    first = _attrs(found[0])
    assert float(first["HEIGHT"]) == pytest.approx(8.4) and float(first["CROWN_DIA"]) == pytest.approx(5.0)
    assert float(first["TRUNK_DIA"]) == pytest.approx(0.35)
    assert found[0].dxf.xscale == pytest.approx(2.5)
    assert float(_attrs(found[1])["TRUNK_DIA"]) == pytest.approx(0.22)


def test_tree_without_trunk_diameter_is_assumed_and_na(tmp_path: Path) -> None:
    (element,), _ = veg.build_vegetation_elements([_tree(3, (3.0, 4.0), trunk=None)], [])
    assert element.param_origin["trunk_diameter"] is ParamOrigin.ASSUMED
    assert element.params.trunk_diameter == pytest.approx(0.30)
    entities = veg.to_dxf_entities([element], FRAME)
    _, doc = _dxf(entities, tmp_path)
    (insert,) = [e for e in doc.modelspace() if e.dxftype() == "INSERT"]
    assert _attrs(insert)["TRUNK_DIA"] == "n/a"
    assert _attrs(insert)["CROWN_DIA"] != "n/a"


def test_rejected_tree_only_in_qa(tmp_path: Path) -> None:
    trees = [_tree(1, (3.0, 4.0)), _tree(2, (9.0, 9.0), status="rejected"), _tree(3, (5.0, 5.0), height=1.2)]
    elements, qa = veg.build_vegetation_elements(trees, [])
    assert len(elements) == 1 and sorted(q.element_id for q in qa) == ["tree-2", "tree-3"]
    assert all(q.category == "tree" and q.reason for q in qa)
    assert len(veg.to_dxf_entities(elements, FRAME)["S-TREE"]) == 1
    scene = Scene()
    for element in elements:
        scene.add(element)
    write_ifc(scene, tmp_path / "t.ifc")
    import ifcopenshell
    assert len(ifcopenshell.open(str(tmp_path / "t.ifc")).by_type("IfcGeographicElement")) == 1


def test_shrub_element_insert_and_closed_outline(tmp_path: Path) -> None:
    result = veg.build_vegetation([], [_shrub()], dtm_sampler=_dtm)
    (element,) = result.elements
    assert isinstance(element, ShrubElement)
    layers = veg.to_dxf_entities(result.elements, FRAME, _dtm)
    assert [e["type"] for e in layers["S-SHRUB"]] == ["insert", "lwpolyline"]
    _, doc = _dxf(layers, tmp_path)
    (insert,) = [e for e in doc.modelspace() if e.dxftype() == "INSERT"]
    assert insert.dxf.name == "SHRUB" and float(_attrs(insert)["HEIGHT"]) == pytest.approx(1.37)
    (poly,) = [e for e in doc.modelspace() if e.dxftype() == "LWPOLYLINE"]
    assert poly.closed and len(poly) == 4 and poly.dxf.layer == "S-SHRUB"
    expected = 430.0 + element.params.base_z
    assert poly.dxf.elevation == pytest.approx(expected)


def test_lawn_polygon_closed_lwpolyline_no_3dface(tmp_path: Path) -> None:
    result = veg.build_vegetation([], [], [_lawn()], _dtm)
    assert result.elements == [] and len(result.lawn_areas) == 1
    entities = veg.to_dxf_entities([], FRAME, _dtm, result.lawn_areas)
    _, doc = _dxf(entities, tmp_path)
    kinds = [e.dxftype() for e in doc.modelspace() if e.dxf.layer == "S-VEG"]
    assert kinds == ["LWPOLYLINE"]
    (poly,) = [e for e in doc.modelspace() if e.dxf.layer == "S-VEG"]
    assert poly.closed
    assert poly.dxf.elevation == pytest.approx(430.0 + result.lawn_areas[0].elevation)
    assert not [e for e in doc.modelspace() if e.dxftype() == "3DFACE"]


def test_steep_lawn_is_closed_3d_polyline(tmp_path: Path) -> None:
    result = veg.build_vegetation([], [], [_lawn(40.0)], lambda xy: 0.05 * xy[:, 0])
    assert result.lawn_areas[0].z_range > 0.3
    _, doc = _dxf(veg.to_dxf_entities([], FRAME, None, result.lawn_areas), tmp_path)
    (poly,) = [e for e in doc.modelspace() if e.dxf.layer == "S-VEG"]
    assert poly.dxftype() == "POLYLINE" and poly.is_closed and poly.is_3d_polyline


def test_ifc_crown_triangles_and_qa_cad_dxf_rules(tmp_path: Path) -> None:
    result = veg.build_vegetation(
        [_tree(1, (3.0, 4.0)), _tree(2, (7.5, 2.2), trunk=None)], [_shrub()], [_lawn()], _dtm)
    scene = Scene()
    for element in result.elements:
        scene.add(element)
    ifc_path = tmp_path / "veg.ifc"
    write_ifc(scene, ifc_path)
    import ifcopenshell
    ifc = ifcopenshell.open(str(ifc_path))
    rule7 = qa_cad._rule_trees(ifc, {})
    assert rule7["ok"], rule7
    entities = veg.to_dxf_entities(result.elements, FRAME, _dtm, result.lawn_areas)
    assert set(entities) == {"S-TREE", "S-SHRUB", "S-VEG"}
    path, _ = _dxf(entities, tmp_path)
    by_rule = {r["rule"]: r for r in qa_cad.check_dxf(path)}
    for name in ("1_block_inserts", "2_area_layers_closed_polylines"):
        assert by_rule[name]["ok"], by_rule[name]


def test_category_counts() -> None:
    trees = [_tree(1, (3.0, 4.0)), _tree(2, (9.0, 9.0), status="rejected"), _tree(4, (6.0, 1.0))]
    result = veg.build_vegetation(trees, [_shrub(0), _shrub(1)], [_lawn(), _lawn(8.0)], _dtm)
    assert veg.category_counts(result.elements, result.lawn_areas) == {"tree": 2, "shrub": 2, "vegetation_area": 2}


def test_shrub_ifc_min_z_equals_dtm_and_nan_dtm_is_qa(tmp_path: Path) -> None:
    result = veg.build_vegetation([], [_shrub()], dtm_sampler=_dtm)
    (element,) = result.elements
    assert element.param_origin["base_z"] is ParamOrigin.MEASURED
    scene = Scene()
    scene.add(element)
    write_ifc(scene, tmp_path / "s.ifc")
    import ifcopenshell
    import ifcopenshell.geom
    ifc = ifcopenshell.open(str(tmp_path / "s.ifc"))
    (shrub,) = ifc.by_type("IfcGeographicElement")
    shape = ifcopenshell.geom.create_shape(ifcopenshell.geom.settings(), shrub)
    z = np.asarray(shape.geometry.verts).reshape(-1, 3)[:, 2]
    assert z.min() == pytest.approx(element.params.base_z, abs=0.01)
    assert z.min() == pytest.approx(float(np.median(_dtm(np.asarray(element.params.outline)))), abs=0.05)
    nan_result = veg.build_vegetation([], [_shrub()], dtm_sampler=lambda xy: np.full(len(xy), np.nan))
    assert nan_result.elements == [] and nan_result.qa[0].reason == "no terrain height"
