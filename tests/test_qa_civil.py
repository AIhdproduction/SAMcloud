"""qa_cad rules for fences, site walls, curbs and gutter stones (counts, fence top z, curb body height)."""
from __future__ import annotations

import ezdxf
import numpy as np

from samcloud.bim.core.ifc import write_ifc
from samcloud.bim.core.model import Scene
from samcloud.drone import contracts as c
from samcloud.drone import qa_cad
from samcloud.dxf.las_source import Frame
from samcloud.dxf.result import BackgroundImage, DxfPlanResult
from samcloud.dxf.writer import build_dxf_document
from samcloud.outdoor_model import civil, site_walls

E0, N0, H0 = 2_600_000.0, 1_200_000.0, 430.0
FRAME = c.LocalFrame(origin_lv95=(E0, N0, H0))
CFG = c.load_drone_config()


def _build(tmp_path, *, dxf_lift_fence=0.0, dxf_lift_curb=0.0):
    fence = site_walls.FenceFeature(0, np.array([[0, 5, 1.2], [4, 5, 1.3], [8, 5, 1.5]], float), 1.0, c.Evidence(points=50),
                                    bottom_z=np.array([0.2, 0.3, 0.5]))
    wall = site_walls.SiteWallFeature(0, np.array([[0, 8], [6, 8], [6, 12]], float), 0.3, 0.0, 1.5, c.Evidence(points=80))
    elements, _ = site_walls.build_site_elements([wall], [fence], config=CFG)
    axis = c.CurbAxis(0, np.array([[0, 0, 0.12], [5, 0, 0.12], [10, 0, 0.13]], float), c.Evidence(points=100),
                      foot_z=np.array([0.0, 0.0, 0.01]), road_side=-1, gutter_width=0.3, gutter_measured=True)
    flat = c.CurbAxis(1, np.array([[0, 20, 0.0], [10, 20, 0.0]], float), c.Evidence(points=100),
                      foot_z=np.array([0.0, 0.0]), road_side=0)
    curbs, _ = civil.build_civil_elements([axis, flat], config=CFG)
    elements = elements + curbs
    counts = {**civil.category_counts(elements)}
    entities = {**site_walls.to_dxf_entities(elements, FRAME), **civil.to_dxf_entities(elements, FRAME, config=CFG)}
    for item in entities.get("S-FENC", []):
        for p in item["points"]:
            p[2] += dxf_lift_fence
    for item in entities.get("S-CURB", []):
        for p in item["points"]:
            p[2] += dxf_lift_curb
    result = DxfPlanResult("t", "site", "outdoor", None, 1.0, Frame(np.zeros(3)), "t", entities, {},
                           dict.fromkeys(entities, 7), BackgroundImage(None, None, (E0, N0, E0 + 20, N0 + 20)))
    dxf = tmp_path / "q.dxf"
    build_dxf_document(result).saveas(dxf)
    scene = Scene()
    scene.settings.origin_offset = (E0, N0, H0)
    for el in elements:
        scene.add(el)
    ifc = tmp_path / "q.ifc"
    write_ifc(scene, ifc)
    return counts, dxf, ifc


def test_counts_include_gutter_fence_and_wall_and_geometry_rules_pass(tmp_path):
    counts, dxf, ifc = _build(tmp_path)
    assert counts == {"curb": 2, "gutter": 1, "marking": 0, "manhole": 0, "fence": 1, "site_wall": 1}
    results = qa_cad.count_consistency(counts, dxf, ifc)
    by_rule = {r["rule"]: r for r in results}
    for category in ("curb", "gutter", "fence", "site_wall"):
        assert by_rule[f"15_count_{category}"]["ok"], by_rule[f"15_count_{category}"]
    assert by_rule["15_count_gutter"]["value"] == {"scene": 1, "dxf": 1, "ifc": 1}
    assert by_rule["16_fence_top_ifc_vs_dxf"]["ok"], by_rule["16_fence_top_ifc_vs_dxf"]
    assert by_rule["16_fence_top_ifc_vs_dxf"]["value"]["vertices"] == 3  # really compared, not skipped
    assert by_rule["17_curb_body_below_axis"]["ok"], by_rule["17_curb_body_below_axis"]
    assert by_rule["17_curb_body_below_axis"]["value"]["curbs"] == 2


def test_s_gutr_is_a_known_outdoor_layer(tmp_path):
    _counts, dxf, _ifc = _build(tmp_path)
    assert "S-GUTR" in ezdxf.readfile(str(dxf)).layers
    assert qa_cad.DXF_CATEGORY_LAYERS["gutter"] == "S-GUTR"


def test_fence_top_mismatch_between_ifc_and_dxf_is_reported(tmp_path):
    counts, dxf, ifc = _build(tmp_path, dxf_lift_fence=0.10)
    rule = {r["rule"]: r for r in qa_cad.count_consistency(counts, dxf, ifc)}["16_fence_top_ifc_vs_dxf"]
    assert not rule["ok"] and rule["offenders"]


def test_curb_body_above_its_dxf_axis_is_reported(tmp_path):
    counts, dxf, ifc = _build(tmp_path, dxf_lift_curb=-0.05)  # the DXF axis lies 5 cm below the IFC body top
    rule = {r["rule"]: r for r in qa_cad.count_consistency(counts, dxf, ifc)}["17_curb_body_below_axis"]
    assert not rule["ok"] and rule["offenders"]
