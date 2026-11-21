"""Tests for samcloud.drone.qa_cad (synthetic DXF/IFC fixtures, deterministic)."""

from __future__ import annotations

import json

import ezdxf
import ifcopenshell
import ifcopenshell.api
import numpy as np
import pytest
from shapely.geometry import Polygon, box

from samcloud.drone import qa_cad

E0, N0 = 2_600_000.0, 1_200_000.0


def _by_rule(results, prefix):
    return [r for r in results if r["rule"].startswith(prefix)]


def _ok(results, prefix):
    found = _by_rule(results, prefix)
    assert found, prefix
    return all(r["ok"] for r in found)


# ---------------------------------------------------------------------------------- DXF fixtures


def _dxf(path, *, tree="insert", pave="lwpoly", contour="spline", curb="thin", frozen=True, labelled=True):
    doc = ezdxf.new("R2018")
    for name in ("S-TREE", "S-SHRUB", "S-MANH", "S-PAVE", "S-TOPO-TIN", "S-TOPO-MAJR", "S-TOPO-TEXT", "S-CURB"):
        doc.layers.add(name)
    if frozen:
        doc.layers.get("S-TOPO-TIN").freeze()
    for block, attrs in (("TREE", ("HEIGHT", "CROWN_DIA", "TRUNK_DIA")), ("SHRUB", ("HEIGHT",)), ("MANHOLE", ("Z_DECKEL", "TYP"))):
        blk = doc.blocks.new(block)
        blk.add_circle((0, 0), 0.5)
        for i, tag in enumerate(attrs):
            blk.add_attdef(tag, (0, -0.2 * i), height=0.1)
    msp = doc.modelspace()
    if tree == "insert":
        ins = msp.add_blockref("TREE", (E0 + 3.3, N0 + 2.7), dxfattribs={"layer": "S-TREE"})
        ins.add_auto_attribs({"HEIGHT": "8.2", "CROWN_DIA": "5.1", "TRUNK_DIA": "0.31"})
    else:
        msp.add_circle((E0 + 3.3, N0 + 2.7), 2.0, dxfattribs={"layer": "S-TREE"})
    ins = msp.add_blockref("SHRUB", (E0 + 5.1, N0 + 1.3), dxfattribs={"layer": "S-SHRUB"})
    ins.add_auto_attribs({"HEIGHT": "1.4"})
    ins = msp.add_blockref("MANHOLE", (E0 + 7.7, N0 + 4.1), dxfattribs={"layer": "S-MANH"})
    ins.add_auto_attribs({"Z_DECKEL": "432.17", "TYP": "Abwasser"})
    pts = [(E0 + 0.13, N0 + 0.21), (E0 + 10.37, N0 + 0.19), (E0 + 10.41, N0 + 6.23), (E0 + 0.17, N0 + 6.27)]
    msp.add_lwpolyline(pts, close=True, dxfattribs={"layer": "S-PAVE"})
    if pave == "3dface":
        msp.add_3dface([(E0, N0, 1.0), (E0 + 1.3, N0, 1.1), (E0, N0 + 1.7, 1.2)], dxfattribs={"layer": "S-PAVE"})
    msp.add_3dface([(E0, N0, 1.0), (E0 + 1.3, N0, 1.1), (E0, N0 + 1.7, 1.2)], dxfattribs={"layer": "S-TOPO-TIN"})
    ctr = [(E0 + 1.1, N0 + 1.2, 432.0), (E0 + 4.3, N0 + 2.9, 432.0), (E0 + 8.9, N0 + 3.1, 432.0), (E0 + 12.2, N0 + 5.7, 432.0)]
    if contour == "spline":
        msp.add_spline(fit_points=ctr, dxfattribs={"layer": "S-TOPO-MAJR"})
    else:
        msp.add_lwpolyline([(x, y) for x, y, _ in ctr], dxfattribs={"layer": "S-TOPO-MAJR"})
    if labelled:
        label = msp.add_text("432.0", height=0.25, dxfattribs={"layer": "S-TOPO-TEXT"})
        label.set_placement((E0 + 4.3, N0 + 2.9), align=ezdxf.enums.TextEntityAlignment.MIDDLE_CENTER)
    if curb == "thin":
        msp.add_lwpolyline([(E0, N0), (E0 + 12.0, N0 + 0.1), (E0 + 14.0, N0 + 3.1), (E0 + 14.2, N0 + 9.4)], dxfattribs={"layer": "S-CURB"})
    else:
        xs = np.arange(0.0, 6.0, 0.02)
        msp.add_lwpolyline([(E0 + x, N0 + 0.1 * x) for x in xs], dxfattribs={"layer": "S-CURB"})
    doc.saveas(path)
    return path


def test_good_dxf_passes_all_rules(tmp_path):
    results = qa_cad.check_dxf(_dxf(tmp_path / "good.dxf"))
    assert {r["rule"][:2] for r in results} >= {"1_", "2_", "3_", "4_", "5_", "6_"}
    assert [r for r in results if not r["ok"]] == []


def _add_shrub_outline(path, closed):
    doc = ezdxf.readfile(path)
    pts = [(E0 + 4.1, N0 + 0.3), (E0 + 6.2, N0 + 0.4), (E0 + 6.1, N0 + 2.3)]
    doc.modelspace().add_lwpolyline(pts, close=closed, dxfattribs={"layer": "S-SHRUB"})
    doc.saveas(path)
    return path


def test_shrub_insert_plus_closed_outline_passes_rule_1(tmp_path):
    path = _add_shrub_outline(_dxf(tmp_path / "shrub.dxf"), closed=True)
    assert _ok(qa_cad.check_dxf(path), "1_")


def test_open_polyline_on_shrub_fails_rule_1(tmp_path):
    path = _add_shrub_outline(_dxf(tmp_path / "shrub_open.dxf"), closed=False)
    results = qa_cad.check_dxf(path)
    assert not _ok(results, "1_")
    assert "LWPOLYLINE" in _by_rule(results, "1_")[0]["offenders"][0]


def test_shrub_count_counts_only_inserts(tmp_path):
    path = _add_shrub_outline(_dxf(tmp_path / "shrub_cnt.dxf"), closed=True)
    msp = ezdxf.readfile(path).modelspace()
    assert qa_cad._count_dxf(msp, "shrub") == 1


def test_tree_as_circle_fails_rule_1(tmp_path):
    results = qa_cad.check_dxf(_dxf(tmp_path / "bad.dxf", tree="circle"))
    assert not _ok(results, "1_")
    assert "CIRCLE" in _by_rule(results, "1_")[0]["offenders"][0]


def test_3dface_on_pave_fails_rules_2_and_3(tmp_path):
    results = qa_cad.check_dxf(_dxf(tmp_path / "bad.dxf", pave="3dface"))
    assert not _ok(results, "2_")
    assert not _ok(results, "3_")


def test_unfrozen_tin_layer_fails_rule_3(tmp_path):
    assert not _ok(qa_cad.check_dxf(_dxf(tmp_path / "bad.dxf", frozen=False)), "3_")


def test_contour_polyline_fails_rule_4(tmp_path):
    assert not _ok(qa_cad.check_dxf(_dxf(tmp_path / "bad.dxf", contour="poly")), "4_")


def test_dense_collinear_curb_fails_rule_5(tmp_path):
    results = qa_cad.check_dxf(_dxf(tmp_path / "bad.dxf", curb="dense"))
    assert not _ok(results, "5_curb_collinear")
    assert not _ok(results, "5_curb_vertex_density")
    assert _by_rule(results, "5_curb_collinear")[0]["value"] > 0.9


def test_local_coordinates_fail_rule_6_unless_insbase(tmp_path):
    doc = ezdxf.readfile(_dxf(tmp_path / "g.dxf"))
    for e in doc.modelspace().query("LWPOLYLINE"):
        e.translate(-E0, -N0, 0)
    bad = tmp_path / "local.dxf"
    doc.saveas(bad)
    assert not _ok(qa_cad.check_dxf(bad), "6_")
    cfg = {"dxf": {"local_origin": True}}
    assert not _ok(qa_cad.check_dxf(bad, cfg), "6_")
    doc.header["$INSBASE"] = (E0, N0, 0.0)
    doc.saveas(bad)
    assert _ok(qa_cad.check_dxf(bad, cfg), "6_")


# ---------------------------------------------------------------------------------- IFC fixtures


def _extrusion(f, ctx, x0, y0, z0, dx, dy, dz):
    """Axis-aligned box as IfcExtrudedAreaSolid representation with its lower corner at (x0, y0, z0)."""
    pts = [f.createIfcCartesianPoint(p) for p in ((0.0, 0.0), (dx, 0.0), (dx, dy), (0.0, dy), (0.0, 0.0))]
    profile = f.createIfcArbitraryClosedProfileDef("AREA", None, f.createIfcPolyline(pts))
    place = f.createIfcAxis2Placement3D(f.createIfcCartesianPoint((x0, y0, z0)), None, None)
    solid = f.createIfcExtrudedAreaSolid(profile, place, f.createIfcDirection((0.0, 0.0, 1.0)), dz)
    return solid


def _tessellation(f, verts, tris):
    pl = f.createIfcCartesianPointList3D([tuple(float(c) for c in v) for v in verts])
    return f.createIfcTriangulatedFaceSet(pl, None, None, [tuple(int(i) + 1 for i in t) for t in tris], None)


def _assign(f, ctx, element, items, rep_type):
    rep = f.createIfcShapeRepresentation(ctx, "Body", rep_type, items)
    ifcopenshell.api.geometry.assign_representation(f, product=element, representation=rep)


def _new_ifc(*, georef=True):
    f = ifcopenshell.api.run("project.create_file", version="IFC4")
    ifcopenshell.api.run("root.create_entity", f, ifc_class="IfcProject", name="qa")
    ifcopenshell.api.run("unit.assign_unit", f, length={"is_metric": True, "raw": "METERS"})
    ctx = ifcopenshell.api.run("context.add_context", f, context_type="Model")
    body = ifcopenshell.api.run("context.add_context", f, context_type="Model", context_identifier="Body", target_view="MODEL_VIEW", parent=ctx)
    if georef:
        ifcopenshell.api.run("georeference.add_georeferencing", f)
        ifcopenshell.api.run("georeference.edit_georeferencing", f,
                             projected_crs={"Name": "EPSG:2056", "GeodeticDatum": "CH1903+"},
                             coordinate_operation={"Eastings": E0, "Northings": N0, "OrthogonalHeight": 400.0})
    return f, body


def _blob(n, seed=1):
    """Closed-ish triangle soup with n triangles around a centre (deterministic noise)."""
    rng = np.random.default_rng(seed)
    verts = rng.normal(scale=1.3, size=(n + 2, 3)) + (0.0, 0.0, 6.0)
    tris = [(i, (i + 1) % (n + 2), (i + 2) % (n + 2)) for i in range(n)]
    return verts, tris


def _add_tree(f, body, n_tris, x=3.3, y=2.7):
    tree = ifcopenshell.api.run("root.create_entity", f, ifc_class="IfcGeographicElement", name="Baum", predefined_type="USERDEFINED")
    verts, tris = _blob(n_tris)
    trunk = _extrusion(f, body, x - 0.15, y - 0.15, 0.0, 0.3, 0.3, 4.0)
    _assign(f, body, tree, [trunk, _tessellation(f, verts + (x, y, 0.0), tris)], "Brep")
    return tree


def _terrain_mesh(z=0.0, hole=None):
    """Regular-ish grid terrain 0..20 m, jittered so centroids never sit on edges."""
    rng = np.random.default_rng(3)
    n = 9
    xs = np.linspace(0.0, 20.0, n)
    verts = [(x + rng.uniform(-0.2, 0.2) * (0 < i < n - 1), y + rng.uniform(-0.2, 0.2) * (0 < j < n - 1), z + 0.01 * rng.random())
             for i, x in enumerate(xs) for j, y in enumerate(xs)]
    tris = []
    for i in range(n - 1):
        for j in range(n - 1):
            a, b, c, d = i * n + j, (i + 1) * n + j, (i + 1) * n + j + 1, i * n + j + 1
            tris += [(a, b, c), (a, c, d)]
    return np.array(verts), tris


def _add_terrain(f, body, z=0.0):
    terrain = ifcopenshell.api.run("root.create_entity", f, ifc_class="IfcGeographicElement", name="Terrain", predefined_type="TERRAIN")
    verts, tris = _terrain_mesh(z)
    _assign(f, body, terrain, [_tessellation(f, verts, tris)], "Tessellation")
    return terrain


def _add_pavement(f, body, x0=6.0, y0=6.0, dx=5.0, dy=4.0, thick=0.2, name="Strasse"):
    p = ifcopenshell.api.run("root.create_entity", f, ifc_class="IfcCivilElement", name=name)
    _assign(f, body, p, [_extrusion(f, body, x0, y0, -thick, dx, dy, thick)], "SweptSolid")
    return p


def _add_curb(f, body, *, triangulated=False):
    c = ifcopenshell.api.run("root.create_entity", f, ifc_class="IfcCivilElement", name="Randstein")
    if triangulated:
        v = np.array([(0, 0, 0), (3.1, 0.2, 0), (3.1, 0.2, 0.25), (0.1, 0.3, 0.25)], float)
        item = _tessellation(f, v, [(0, 1, 2), (0, 2, 3)])
    else:
        item = _extrusion(f, body, 1.0, 1.0, 0.0, 3.0, 0.12, 0.25)
    _assign(f, body, c, [item], "Tessellation" if triangulated else "SweptSolid")
    return c


def _add_door(f, body, *, z, fills=True):
    wall = ifcopenshell.api.run("root.create_entity", f, ifc_class="IfcWall", name="Wand")
    _assign(f, body, wall, [_extrusion(f, body, 12.1, 3.3, 0.0, 0.3, 4.0, 3.0)], "SweptSolid")
    door = ifcopenshell.api.run("root.create_entity", f, ifc_class="IfcDoor", name="Tuer")
    _assign(f, body, door, [_extrusion(f, body, 12.2, 4.4, z, 0.1, 1.0, 2.1)], "SweptSolid")
    if fills:
        opening = ifcopenshell.api.run("root.create_entity", f, ifc_class="IfcOpeningElement", name="Oeffnung")
        ifcopenshell.api.run("feature.add_feature", f, feature=opening, element=wall)
        ifcopenshell.api.run("feature.add_filling", f, opening=opening, element=door)
    return door


def _save(f, tmp_path, name="m.ifc"):
    p = tmp_path / name
    f.write(str(p))
    return p


def test_tree_with_200_triangles_fails_rule_7_100_passes(tmp_path):
    f, body = _new_ifc()
    _add_tree(f, body, 100)
    assert _ok(qa_cad.check_ifc(_save(f, tmp_path, "a.ifc")), "7_")
    f2, body2 = _new_ifc()
    _add_tree(f2, body2, 200)
    bad = qa_cad.check_ifc(_save(f2, tmp_path, "b.ifc"))
    assert not _ok(bad, "7_")
    assert "200" in _by_rule(bad, "7_")[0]["offenders"][0]
    f3, body3 = _new_ifc()
    _add_tree(f3, body3, 20)
    assert not _ok(qa_cad.check_ifc(_save(f3, tmp_path, "c.ifc")), "7_")


def test_curb_triangulated_fails_rule_8(tmp_path):
    f, body = _new_ifc()
    _add_curb(f, body)
    assert _ok(qa_cad.check_ifc(_save(f, tmp_path, "a.ifc")), "8_")
    f2, body2 = _new_ifc()
    _add_curb(f2, body2, triangulated=True)
    assert not _ok(qa_cad.check_ifc(_save(f2, tmp_path, "b.ifc")), "8_")


def test_pavement_thickness_rule_9(tmp_path):
    f, body = _new_ifc()
    _add_pavement(f, body, thick=0.2)
    assert _ok(qa_cad.check_ifc(_save(f, tmp_path, "a.ifc")), "9_")
    f2, body2 = _new_ifc()
    _add_pavement(f2, body2, thick=0.03)
    assert not _ok(qa_cad.check_ifc(_save(f2, tmp_path, "b.ifc")), "9_")


def test_terrain_triangle_inside_pavement_fails_rule_10(tmp_path):
    f, body = _new_ifc()
    _add_terrain(f, body)
    _add_pavement(f, body)  # terrain covers the pavement footprint -> centroids inside
    bad = qa_cad.check_ifc(_save(f, tmp_path, "a.ifc"))
    assert not _ok(bad, "10_")
    # footprint elsewhere -> terrain clean
    good = qa_cad.check_ifc(tmp_path / "a.ifc", pavement_footprints=[box(100.0, 100.0, 105.0, 104.0)])
    assert _ok(good, "10_")


def test_missing_map_conversion_fails_rule_11(tmp_path):
    f, body = _new_ifc(georef=False)
    _add_tree(f, body, 100)
    bad = qa_cad.check_ifc(_save(f, tmp_path, "a.ifc"))
    assert not _ok(bad, "11_")
    f2, body2 = _new_ifc()
    _add_tree(f2, body2, 100)
    assert _ok(qa_cad.check_ifc(_save(f2, tmp_path, "b.ifc")), "11_")


def test_floating_door_fails_rule_12(tmp_path):
    f, body = _new_ifc()
    _add_terrain(f, body)
    _add_door(f, body, z=0.05)
    assert _ok(qa_cad.check_ifc(_save(f, tmp_path, "a.ifc")), "12_")
    f2, body2 = _new_ifc()
    _add_terrain(f2, body2)
    _add_door(f2, body2, z=1.2)
    bad = qa_cad.check_ifc(_save(f2, tmp_path, "b.ifc"))
    assert not _ok(bad, "12_")
    f3, body3 = _new_ifc()
    _add_terrain(f3, body3)
    _add_door(f3, body3, z=0.0, fills=False)
    assert not _ok(qa_cad.check_ifc(_save(f3, tmp_path, "c.ifc")), "12_")


def test_ids_rule_reports_counts(tmp_path):
    f, body = _new_ifc()
    _add_tree(f, body, 100)
    ids_rule = _by_rule(qa_cad.check_ifc(_save(f, tmp_path, "a.ifc")), "13_")[0]
    assert set(ids_rule["value"]) == {"pass", "fail"}


# ---------------------------------------------------------------------------------- topology / counts / report


def test_topology_gap_and_overlap(tmp_path):
    a = box(0.0, 0.0, 1.0, 1.0)
    gap2 = box(1.0 + 0.0002, 0.0, 2.0, 1.0)      # 1 m x 0.2 mm = 2 cm2
    gap05 = box(1.0 + 0.00005, 0.0, 2.0, 1.0)    # 0.5 cm2
    bad = qa_cad.check_topology({"road": [a], "sidewalk": [gap2]})
    assert not _ok(bad, "14_topology_gap_overlap")
    assert 1.9 < _by_rule(bad, "14_topology_gap")[0]["value"] < 2.1
    good = qa_cad.check_topology({"road": [a], "sidewalk": [gap05]})
    assert _ok(good, "14_")
    overlap = qa_cad.check_topology({"road": [a], "sidewalk": [box(0.9998, 0.0, 2.0, 1.0)]})
    assert not _ok(overlap, "14_topology_gap_overlap")


def test_topology_invalid_polygon_fails():
    bow = Polygon([(0, 0), (1, 1), (1, 0), (0, 1)])
    assert not _ok(qa_cad.check_topology({"road": [bow]}), "14_topology_valid")


def test_count_consistency_detects_mismatch(tmp_path):
    dxf = _dxf(tmp_path / "a.dxf")
    f, body = _new_ifc()
    _add_tree(f, body, 100)
    _add_terrain(f, body)
    ifc = _save(f, tmp_path, "a.ifc")
    ok = [r for r in qa_cad.count_consistency({"tree": 1}, dxf, ifc) if r["rule"].startswith("15_")]  # the geometry rules 16-19 need matching scenes
    assert all(r["ok"] for r in ok), ok
    bad = qa_cad.count_consistency({"tree": 2, "shrub": 1}, dxf, ifc)
    by = {r["rule"]: r for r in bad}
    assert not by["15_count_tree"]["ok"]
    assert by["15_count_tree"]["value"] == {"scene": 2, "dxf": 1, "ifc": 1}


def test_write_qa_report(tmp_path):
    results = qa_cad.check_dxf(_dxf(tmp_path / "bad.dxf", tree="circle"))
    jp, mp = qa_cad.write_qa_report(results, tmp_path / "out")
    data = json.loads(jp.read_text(encoding="utf-8"))
    assert len(data) == len(results) and {"rule", "ok", "value", "threshold", "offenders"} <= set(data[0])
    md = mp.read_text(encoding="utf-8")
    assert "| Regel | OK | Wert | Grenze | Beispiele |" in md and "NEIN" in md


def _real_outdoor_scene():
    from samcloud.bim.core.model import (
        BuildingElement, BuildingParams, CurbElement, CurbParams, FenceElement, FenceParams, ManholeElement,
        ManholeParams, MarkingElement, MarkingParams, OpeningElement, OpeningParams, PavementElement,
        PavementParams, Scene, ShrubElement, ShrubParams, SiteWallElement, SiteWallParams, TerrainElement,
        TerrainParams, TreeElement, TreeParams, VegetationAreaElement, VegetationAreaParams, WallElement,
        WallFunction, WallParams, WindowElement, WindowParams,
    )
    scene = Scene()
    scene.settings.origin_offset = (2600000.0, 1200000.0, 500.0)
    verts = [(0, 0, 0), (2, 0, 0), (2, 2, 0), (0, 2, 0), (0, 0, 2), (2, 0, 2), (2, 2, 2), (0, 2, 2)]
    tris = [(0, 1, 5), (0, 5, 4), (1, 2, 6), (1, 6, 5), (2, 3, 7), (2, 7, 6), (3, 0, 4), (3, 4, 7), (4, 5, 6), (4, 6, 7)]
    building = BuildingElement(params=BuildingParams(vertices=verts, triangles=tris, face_roles=[1] * 8 + [2, 2]))
    scene.add(building)
    wall = WallElement(building_id=building.id, params=WallParams(start=(0, 0), end=(2, 0), thickness=.2, height=2, function=WallFunction.EXTERIOR))
    scene.add(wall)
    opening = OpeningElement(params=OpeningParams(host_wall_id=wall.id, position_along_axis=.4, width=.6, height=1))
    scene.add(opening)
    scene.add(WindowElement(params=WindowParams(opening_id=opening.id)))
    scene.add(TerrainElement(params=TerrainParams(vertices=[(-20, -20, 0), (40, -20, 0), (-20, 40, 0)], triangles=[(0, 1, 2)])))
    scene.add(TreeElement(params=TreeParams(trunk_xy=(10, 2), base_z=0, height=6, crown_diameter=4)))
    scene.add(TreeElement(params=TreeParams(trunk_xy=(14, 2), base_z=0, height=5, crown_diameter=3)))
    scene.add(ShrubElement(params=ShrubParams(outline=[(20, 0), (21, 0), (20, 1)], base_z=0, height=1)))
    scene.add(VegetationAreaElement(params=VegetationAreaParams(outline=[(20, 7), (21, 7), (20, 8)], base_z=0)))
    scene.add(SiteWallElement(params=SiteWallParams(axis=[(0, 5), (4, 5)], thickness=.2, base_z=0)))
    scene.add(FenceElement(params=FenceParams(axis=[(0, 6, 0), (4, 6, 0)], height=1)))
    scene.add(PavementElement(params=PavementParams(vertices=[(6, 6, 0), (11, 6, 0), (6, 10, 0)], triangles=[(0, 1, 2)], thickness=0.2)))
    scene.add(CurbElement(params=CurbParams(axis=[(0, 8, 0), (3, 8, 0)])))
    scene.add(MarkingElement(params=MarkingParams(points=[(0, 9), (2, 9)])))
    scene.add(ManholeElement(params=ManholeParams(center=(5, 3, 0), diameter=.6)))
    return scene


def test_real_writer_outdoor_scene_rules_and_counts(tmp_path):
    from samcloud.bim.core.ifc import write_ifc
    ifc_path = tmp_path / "real.ifc"
    write_ifc(_real_outdoor_scene(), ifc_path)
    results = qa_cad.check_ifc(ifc_path)
    for prefix in ("7_", "8_", "9_", "11_"):
        assert _ok(results, prefix), _by_rule(results, prefix)
    assert _by_rule(results, "7_")[0]["value"]["trees"] == 2
    assert _by_rule(results, "8_")[0]["value"] == 1
    assert _by_rule(results, "9_")[0]["value"]["pavements"] == 1
    counts = {"tree": 2, "shrub": 1, "vegetation_area": 1, "site_wall": 1, "fence": 1, "marking": 1, "manhole": 1,
              "curb": 1, "road": 1, "window": 1, "building": 1}
    doc = ezdxf.new("R2018")
    msp = doc.modelspace()
    for category, n in counts.items():
        layer = qa_cad.DXF_CATEGORY_LAYERS[category]
        if layer not in doc.layers:
            doc.layers.add(layer)
        for i in range(n):
            if category == "shrub":
                if "SHRUB" not in doc.blocks:
                    doc.blocks.new("SHRUB")
                msp.add_blockref("SHRUB", (float(i), 0.0), dxfattribs={"layer": layer})
            elif category == "vegetation_area":
                msp.add_lwpolyline([(0, 0), (1, 0), (1, 1)], close=True, dxfattribs={"layer": layer})
            else:
                msp.add_point((float(i), 0.0), dxfattribs={"layer": layer})
    dxf_path = tmp_path / "real.dxf"
    doc.saveas(dxf_path)
    out = qa_cad.count_consistency(counts, dxf_path, ifc_path)
    assert all(r["ok"] for r in out), [r for r in out if not r["ok"]]


def test_count_dxf_pavement_hole_not_counted(tmp_path):
    doc = ezdxf.new("R2018")
    doc.layers.add("S-PAVE")
    msp = doc.modelspace()
    outer = [(E0 + 0.13, N0 + 0.21), (E0 + 10.37, N0 + 0.19), (E0 + 10.41, N0 + 6.23), (E0 + 0.17, N0 + 6.27)]
    hole = [(E0 + 3.11, N0 + 2.13), (E0 + 5.37, N0 + 2.19), (E0 + 5.41, N0 + 4.23)]
    other = [(E0 + 20.1, N0 + 0.3), (E0 + 24.2, N0 + 0.4), (E0 + 24.1, N0 + 3.3)]
    for pts in (outer, hole, other):
        msp.add_lwpolyline(pts, close=True, dxfattribs={"layer": "S-PAVE"})
    assert qa_cad._count_dxf(msp, "road") == 2
    solo = ezdxf.new("R2018")
    solo.layers.add("S-PAVE")
    solo.modelspace().add_lwpolyline(outer, close=True, dxfattribs={"layer": "S-PAVE"})
    solo.modelspace().add_lwpolyline(hole, close=True, dxfattribs={"layer": "S-PAVE"})
    assert qa_cad._count_dxf(solo.modelspace(), "road") == 1


def test_unlabelled_long_major_contour_fails_rule_15(tmp_path):
    results = qa_cad.check_dxf(_dxf(tmp_path / "nolabel.dxf", labelled=False))
    assert not _ok(results, "15_contours_labelled")
    assert _ok(qa_cad.check_dxf(_dxf(tmp_path / "label.dxf")), "15_contours_labelled")


def _add_tree_stem(path):
    doc = ezdxf.readfile(path)
    blk = doc.blocks.new("TREE_STEM")
    blk.add_circle((0, 0), 0.5)
    doc.modelspace().add_blockref("TREE_STEM", (E0 + 3.3, N0 + 2.7), dxfattribs={"layer": "S-TREE"})
    doc.saveas(path)


def test_tree_stem_insert_allowed_and_not_counted_as_tree(tmp_path):
    dxf = _dxf(tmp_path / "stem.dxf")
    _add_tree_stem(dxf)
    assert _ok(qa_cad.check_dxf(dxf), "1_block_inserts")
    f, body = _new_ifc()
    _add_tree(f, body, 100)
    _add_terrain(f, body)
    ifc = _save(f, tmp_path, "stem.ifc")
    result = [r for r in qa_cad.count_consistency({"tree": 1}, dxf, ifc) if r["rule"].startswith("15_")]
    assert all(r["ok"] for r in result), result
    assert result[0]["value"]["dxf"] == 1
