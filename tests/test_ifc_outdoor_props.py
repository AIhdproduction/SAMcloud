"""Outdoor IFC attributes: SAMcloud_<Category> Pset + Qto_SAMcloud_<Category> per product, LV95/LN02, props.json, QA rules."""
from __future__ import annotations

import json
import math

import ezdxf
import ifcopenshell
import ifcopenshell.util.element as element_util
import pytest

from samcloud.bim.core.ifc import write_ifc
from samcloud.bim.core.ifc.outdoor import outdoor_category_of_product
from samcloud.bim.core.model import (
    BuildingElement, BuildingParams, CurbElement, CurbParams, FenceElement, FenceParams, ManholeElement, ManholeParams,
    MarkingElement, MarkingParams, ParamOrigin, PavementElement, PavementParams, Scene, ShrubElement, ShrubParams,
    SiteWallElement, SiteWallParams, TerrainElement, TerrainParams, TreeElement, TreeParams, VegetationAreaElement,
    VegetationAreaParams,
)
from samcloud.drone import qa_cad

OFFSET = (2_600_000.0, 1_200_000.0, 430.0)


def _rect(x0, y0, x1, y1, z=1.0):
    """Two-triangle rectangle with slight height noise."""
    return ([(x0, y0, z), (x1, y0, z + 0.013), (x1, y1, z + 0.004), (x0, y1, z - 0.007)], [(0, 1, 2), (0, 2, 3)])


def _box_building():
    """Closed 10 x 6 m house, eaves 3 m, ridge 4.5 m (gable); roles ground=0 wall=1 roof=2."""
    v = [(0, 0, 0.01), (10, 0, 0.0), (10, 6, 0.02), (0, 6, 0.0),
         (0, 0, 3.0), (10, 0, 3.0), (10, 6, 3.0), (0, 6, 3.0), (0, 3, 4.5), (10, 3, 4.5)]
    t = [(0, 2, 1), (0, 3, 2),
         (0, 1, 5), (0, 5, 4), (1, 2, 6), (1, 6, 5), (2, 3, 7), (2, 7, 6), (3, 0, 4), (3, 4, 7),
         (4, 5, 9), (4, 9, 8), (7, 8, 9), (7, 9, 6), (4, 8, 7), (5, 6, 9)]
    roles = [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 1, 1]
    return v, t, roles


def _scene(*, trunk_measured: bool = False) -> Scene:
    scene = Scene()
    scene.settings.origin_offset = OFFSET
    tv, tt = _rect(0, 0, 10, 5, 1.0)
    scene.add(PavementElement(params=PavementParams(vertices=tv, triangles=tt, kind="road", thickness=0.1),
                              param_origin={"vertices": ParamOrigin.MEASURED}))
    gv, gt = _rect(-5, -5, 30, 30, 0.0)
    scene.add(TerrainElement(params=TerrainParams(vertices=gv, triangles=gt), param_origin={"vertices": ParamOrigin.FITTED}))
    scene.add(ManholeElement(params=ManholeParams(center=(3.0, 2.0, 1.234), diameter=0.6, type="Schacht"),
                             param_origin={"center": ParamOrigin.MEASURED, "diameter": ParamOrigin.FITTED}, evidence={"roundness": 0.9}))
    tree_params = TreeParams(trunk_xy=(15.0, 10.0), base_z=0.5, height=6.0, crown_diameter=4.0, crown_shape="ellipsoid",
                             crown_offset=(0.2, 0.1), trunk_diameter=0.35 if trunk_measured else None, trunk_measured=trunk_measured or None)
    scene.add(TreeElement(params=tree_params, param_origin={"height": ParamOrigin.MEASURED, "crown_diameter": ParamOrigin.MEASURED}))
    scene.add(ShrubElement(params=ShrubParams(outline=[(20, 20), (22, 20), (22, 21), (20, 21)], base_z=0.2, height=1.0)))
    scene.add(VegetationAreaElement(params=VegetationAreaParams(outline=[(12, 0), (20, 0), (20, 4), (12, 4)], base_z=0.1)))
    scene.add(SiteWallElement(params=SiteWallParams(axis=[(0, 8), (6, 8)], thickness=0.3, base_z=0.0, top_z=1.5)))
    scene.add(FenceElement(params=FenceParams(axis=[(0, 12, 1.2), (4, 12, 1.4), (8, 12, 1.2)], height=1.0, bottom_z=[0.2, 0.3, 0.2])))
    scene.add(CurbElement(params=CurbParams(axis=[(0, 20, 0.12), (5, 20, 0.12), (10, 20, 0.13)], foot_z=[0.0, 0.0, 0.01],
                                            road_side=-1, gutter_width=0.3), evidence={"flush": False, "flush_stations": 0, "gutter": "assumed"}))
    scene.add(MarkingElement(params=MarkingParams(points=[(0, 1, 1.0), (10, 1, 1.0)])))
    v, t, roles = _box_building()
    scene.add(BuildingElement(params=BuildingParams(vertices=v, triangles=t, face_roles=roles), evidence={"lod": "2.2", "rmse": 0.12},
                              param_origin={"vertices": ParamOrigin.FITTED}))
    return scene


@pytest.fixture(scope="module")
def exported(tmp_path_factory):
    path = tmp_path_factory.mktemp("props") / "o.ifc"
    write_ifc(_scene(), path)
    return path, ifcopenshell.open(str(path))


def _sets(ifc, category):
    for product in ifc.by_type("IfcProduct"):
        if outdoor_category_of_product(product) == category or (category == "building" and product.is_a("IfcBuilding")):
            return element_util.get_psets(product)
    raise AssertionError(f"no {category}")


def herkunft(ifc, category):
    return _sets(ifc, category)["SAMcloud_Herkunft"]


def _pair(ifc, category, suffix):
    sets = _sets(ifc, category)
    assert f"SAMcloud_{suffix}" in sets and f"Qto_SAMcloud_{suffix}" in sets, list(sets)
    return sets[f"SAMcloud_{suffix}"], sets[f"Qto_SAMcloud_{suffix}"]


def test_every_category_has_pset_and_qto_with_plausible_values(exported):
    _, ifc = exported
    p, q = _pair(ifc, "road", "Pavement")
    assert p["Flaeche_m2"] == pytest.approx(50.0, abs=0.1) and q["Flaeche_m2"] == pytest.approx(50.0, abs=0.1)
    assert p["Oberflaeche_m2"] >= p["Flaeche_m2"] and p["Volumen_m3"] == pytest.approx(5.0, abs=0.01)
    assert p["Umfang_m"] == pytest.approx(30.0, abs=0.1) and p["Aufbaudicke_m"] == 0.1 and "Gefaelle_mittel_%" in p
    assert p["Layer"] == "S-PAVE" and herkunft(ifc, "road")["Source"] == "MEASURED" and {"Hoehe_min_LN02", "Hoehe_mittel_LN02", "Hoehe_max_LN02"} <= set(p)

    p, q = _pair(ifc, "manhole", "Manhole")
    assert p["Deckelflaeche_m2"] == pytest.approx(math.pi * 0.3 ** 2, abs=0.001) and q["Deckelflaeche_m2"] == p["Deckelflaeche_m2"]
    assert {"Durchmesser_m", "Typ", "Rundheit", "Deckelhoehe_LN02", "Hoehe_mittel_LN02"} <= set(p) and p["Layer"] == "S-MANH"

    p, q = _pair(ifc, "tree", "Tree")
    assert {"Hoehe_m", "Kronendurchmesser_m", "Kronenansatz_LN02", "Kronenvolumen_m3", "Fusshoehe_LN02", "Stamm_LV95_E", "Kronenmitte_LV95_N"} <= set(p)
    assert "Kronenform" not in p and "CrownShape" not in p
    assert 0.0 < q["Kronenvolumen_m3"] < (4 / 3) * math.pi * 2.0 ** 3

    p, q = _pair(ifc, "shrub", "Shrub")
    assert p["Grundflaeche_m2"] == pytest.approx(2.0) and q["Volumen_m3"] == pytest.approx(2.0)
    p, q = _pair(ifc, "lawn", "VegetationArea")
    assert p["Flaeche_m2"] == pytest.approx(32.0)
    p, q = _pair(ifc, "site_wall", "SiteWall")
    assert p["Laenge_m"] == pytest.approx(6.0) and p["Dicke_m"] == pytest.approx(0.3) and p["Hoehe_max_m"] == pytest.approx(1.5)
    assert p["Grundflaeche_m2"] == pytest.approx(1.8, abs=0.01) and q["Volumen_m3"] == pytest.approx(2.7, abs=0.02)
    p, q = _pair(ifc, "fence", "Fence")
    assert p["Laenge_m"] == pytest.approx(8.0, abs=0.01) and {"Hoehe_mittel_m", "Hoehe_min_m", "Hoehe_max_m", "Oberkante_max_LN02", "Unterkante_min_LN02"} <= set(p)
    p, q = _pair(ifc, "curb", "Curb")
    assert p["Laenge_m"] == pytest.approx(10.0) and p["Buendig"] is False and p["Wasserstein"] is True and "Anschlag_mittel_m" in p
    assert p["Querschnitt"].startswith("0.120 x")
    p, q = _pair(ifc, "gutter", "Gutter")
    assert p["Flaeche_m2"] == pytest.approx(p["Laenge_m"] * 0.3, abs=0.01) and p["Breite_Herkunft"] == "angenommen"
    p, q = _pair(ifc, "marking", "Marking")
    assert p["Laenge_m"] == pytest.approx(10.0)
    p, q = _pair(ifc, "terrain", "Terrain")
    assert p["Flaeche_m2"] == pytest.approx(35 * 35, abs=1.0) and p["Dreiecke"] == 2 and herkunft(ifc, "terrain")["Source"] == "FITTED"
    p, q = _pair(ifc, "building", "Building")
    assert p["Grundflaeche_m2"] == pytest.approx(60.0, abs=0.1) and p["Dachflaeche_m2"] > 60.0 and p["Fassadenflaeche_m2"] > 100.0
    assert p["Volumen_m3"] == pytest.approx(10 * 6 * 3 + 0.5 * 6 * 1.5 * 10, abs=1.5)
    assert p["Traufhoehe_LN02"] == pytest.approx(433.0, abs=0.01) and p["Firsthoehe_LN02"] == pytest.approx(434.5, abs=0.01)
    assert p["Gebaeudehoehe_m"] == pytest.approx(4.5, abs=0.05) and p["Dachflaechen_Anzahl"] == 2
    assert p["RMSE_Dach_m"] == 0.12 and p["LoD"] == "2.2"
    _pair(ifc, "roof", "Roof")


def test_property_types_are_ifc_typed(exported):
    path, _ = exported
    f = ifcopenshell.open(str(path))
    pset = next(r.RelatingPropertyDefinition for p in f.by_type("IfcDistributionChamberElement") for r in p.IsDefinedBy
                if r.RelatingPropertyDefinition.Name == "SAMcloud_Manhole")
    types = {pr.Name: pr.NominalValue.is_a() for pr in pset.HasProperties}
    assert types["Deckelflaeche_m2"] == "IfcAreaMeasure" and types["Durchmesser_m"] == "IfcLengthMeasure"
    assert types["Typ"] == "IfcLabel" and types["Typ"] == "IfcLabel"
    qto = next(r.RelatingPropertyDefinition for p in f.by_type("IfcDistributionChamberElement") for r in p.IsDefinedBy
               if r.RelatingPropertyDefinition.Name == "Qto_SAMcloud_Manhole")
    assert {q.is_a() for q in qto.Quantities} == {"IfcQuantityLength", "IfcQuantityArea", "IfcQuantityVolume", "IfcQuantityCount"}


def test_tree_trunk_measured_or_not(tmp_path):
    p, _ = _pair(ifcopenshell.open(str(_write(_scene(trunk_measured=False), tmp_path / "a.ifc"))), "tree", "Tree")
    assert p["Stammdurchmesser_m"] == "nicht gemessen" and p["StammGemessen"] is False
    p, q = _pair(ifcopenshell.open(str(_write(_scene(trunk_measured=True), tmp_path / "b.ifc"))), "tree", "Tree")
    assert p["Stammdurchmesser_m"] == pytest.approx(0.35) and p["StammGemessen"] is True and q["Stammdurchmesser_m"] == pytest.approx(0.35)


def _write(scene, path):
    write_ifc(scene, path)
    return path


def test_ln02_and_lv95_use_the_georef_offset(exported):
    _, ifc = exported
    p, _ = _pair(ifc, "manhole", "Manhole")
    assert p["LV95_E"] == pytest.approx(OFFSET[0] + 3.0, abs=1e-3) and p["LV95_N"] == pytest.approx(OFFSET[1] + 2.0, abs=1e-3)
    assert p["Deckelhoehe_LN02"] == pytest.approx(OFFSET[2] + 1.234, abs=0.006)
    t, _ = _pair(ifc, "tree", "Tree")
    assert t["Stamm_LV95_E"] == pytest.approx(OFFSET[0] + 15.0, abs=1e-3)
    assert t["Kronenmitte_LV95_E"] == pytest.approx(OFFSET[0] + 15.2, abs=1e-3) and t["Kronenmitte_LV95_N"] == pytest.approx(OFFSET[1] + 10.1, abs=1e-3)
    assert t["Fusshoehe_LN02"] == pytest.approx(OFFSET[2] + 0.5, abs=1e-3)


def test_props_json_contains_psets_and_quantities(exported, tmp_path):
    from samcloud.bim import viewer_export
    path, _ = exported
    assets = viewer_export.export_viewer_assets(path, tmp_path)
    props = json.loads(assets.props_path.read_text(encoding="utf-8"))
    manholes = [e for e in props["elements"].values() if e["ifc_class"] == "IfcDistributionChamberElement"]
    assert manholes
    psets = manholes[0]["psets"]
    assert psets["SAMcloud_Manhole"]["Deckelflaeche_m2"] > 0 and psets["Qto_SAMcloud_Manhole"]["Durchmesser_m"] == pytest.approx(0.6, abs=0.001)
    trees = [e for e in props["elements"].values() if e["psets"].get("SAMcloud_Tree")]
    assert "Kronenvolumen_m3" in trees[0]["psets"]["Qto_SAMcloud_Tree"]


def _dxf(path, hole_z, crown):
    doc = ezdxf.new("R2018")
    for name, tags in (("MANHOLE", ("Z_DECKEL", "TYP")), ("TREE", ("HEIGHT", "CROWN_DIA", "KRONE"))):
        block = doc.blocks.new(name)
        for tag in tags:
            block.add_attdef(tag, (0, 0))
    for layer in ("S-MANH", "S-TREE"):
        doc.layers.add(layer)
    msp = doc.modelspace()
    msp.add_blockref("MANHOLE", (OFFSET[0] + 3.0, OFFSET[1] + 2.0, hole_z), dxfattribs={"layer": "S-MANH"}).add_auto_attribs(
        {"Z_DECKEL": f"{hole_z:.2f}", "TYP": "Schacht"})
    msp.add_blockref("TREE", (OFFSET[0] + 15.2, OFFSET[1] + 10.1, OFFSET[2] + 0.5), dxfattribs={"layer": "S-TREE"}).add_auto_attribs(
        {"HEIGHT": "6.00", "CROWN_DIA": f"{crown:.2f}", "KRONE": f"KRONE={crown:.2f} m"})
    doc.saveas(path)
    return path


def _rule(results, name):
    return next(r for r in results if r["rule"] == name)


def test_qa_rules_pass_for_consistent_scene_and_fail_for_deviation(exported, tmp_path):
    ifc_path, _ = exported
    good = qa_cad.check_geometry_consistency(_dxf(tmp_path / "g.dxf", OFFSET[2] + 1.234, 4.0), ifc_path)
    assert _rule(good, "20_ifc_attribute_sets")["ok"], _rule(good, "20_ifc_attribute_sets")
    rule = _rule(good, "21_ifc_attributes_vs_dxf")
    assert rule["ok"] and rule["value"]["manhole"] == 1 and rule["value"]["tree"] == 1, rule
    bad_hole = qa_cad.check_geometry_consistency(_dxf(tmp_path / "b1.dxf", OFFSET[2] + 1.30, 4.0), ifc_path)
    assert not _rule(bad_hole, "21_ifc_attributes_vs_dxf")["ok"]
    bad_crown = qa_cad.check_geometry_consistency(_dxf(tmp_path / "b2.dxf", OFFSET[2] + 1.234, 4.5), ifc_path)
    assert not _rule(bad_crown, "21_ifc_attributes_vs_dxf")["ok"]


def test_qa_rule_flags_product_without_attribute_sets(tmp_path):
    scene = Scene()
    scene.settings.origin_offset = OFFSET
    scene.add(ManholeElement(params=ManholeParams(center=(3.0, 2.0, 1.234), diameter=0.6)))
    path = _write(scene, tmp_path / "m.ifc")
    ifc = ifcopenshell.open(str(path))
    for product in ifc.by_type("IfcDistributionChamberElement"):
        for rel in list(product.IsDefinedBy):
            if rel.RelatingPropertyDefinition.Name.startswith("Qto_SAMcloud"):
                ifc.remove(rel.RelatingPropertyDefinition)
                ifc.remove(rel)
    stripped = tmp_path / "s.ifc"
    ifc.write(str(stripped))
    result = qa_cad._rule_ifc_attribute_sets(ifcopenshell.open(str(stripped)))
    assert result is not None and not result["ok"] and "Qto_SAMcloud_Manhole" in result["offenders"][0]


def test_legacy_keys_and_local_values_are_gone_and_provenance_is_separate(exported):
    _, ifc = exported
    legacy = {"Diameter", "Type", "Height", "CrownDiameter", "TrunkDiameter", "TrunkDiameterMeasured", "BaseZ", "TopZ", "Thickness",
              "Width", "Kind", "Sweep", "RoadSide", "Closed", "TriangleCount"}
    for category in ("manhole", "tree", "shrub", "lawn", "site_wall", "fence", "road", "curb", "gutter", "marking", "terrain", "building"):
        sets = _sets(ifc, category)
        main = next(v for k, v in sets.items() if k.startswith("SAMcloud_") and k != "SAMcloud_Herkunft")
        assert not legacy & set(main), (category, legacy & set(main))
        assert not any(k.startswith(("Origin_", "Evidence")) for k in main), category
        assert not {"Source", "Points", "RMSE_m"} & set(main) and "Status" not in main
        assert "Status" in sets["SAMcloud_Herkunft"] and "Source" in sets["SAMcloud_Herkunft"]
    hole = _sets(ifc, "manhole")
    assert hole["SAMcloud_Manhole"]["Z_DECKEL"] == hole["SAMcloud_Manhole"]["Deckelhoehe_LN02"] == pytest.approx(OFFSET[2] + 1.234, abs=0.006)
    assert hole["SAMcloud_Herkunft"]["Herkunft_diameter"] == "fitted" and hole["SAMcloud_Herkunft"]["Eingerastet"] is False
    assert not any("local" in k for k in hole["SAMcloud_Herkunft"])
