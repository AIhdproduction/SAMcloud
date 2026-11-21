"""Crown shape fit, measured trunks, IFC crown meshes and DXF TREE/MANHOLE blocks (synthetic, noisy)."""
from __future__ import annotations

import pickle
from pathlib import Path

import ezdxf
import numpy as np
import pytest

from samcloud.bim.core.ifc import write_ifc
from samcloud.bim.core.model import ParamOrigin, Scene, TreeElement, TreeParams
from samcloud.drone.contracts import TREE, TREE_TRUNK, Evidence, LocalFrame, PointSet, TreeInstance
from samcloud.dxf.las_source import Frame
from samcloud.dxf.result import BackgroundImage, DxfPlanResult
from samcloud.dxf.writer import build_dxf_document, outdoor_symbol_insert
from samcloud.environment.vegetation import detect_trees, fit_crown_shape
from samcloud.outdoor_model import vegetation as veg

FRAME = LocalFrame(origin_lv95=(2_683_000.0, 1_248_000.0, 430.0))
NOISE = 0.03


def _dtm(xy: np.ndarray) -> np.ndarray:
    xy = np.asarray(xy, dtype=float).reshape(-1, 2)
    return 0.02 * xy[:, 0] + 0.01 * xy[:, 1]


# ---- crown generators: surface points of each shape, 3 cm noise ----------------------------------------------------

def _noise(rng, pts):
    return pts + rng.normal(0.0, NOISE, pts.shape)


def sphere(rng, cx, cy, z0, radius=2.1, n=3000):
    v = rng.normal(size=(n, 3))
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    return _noise(rng, np.column_stack([cx + radius * v[:, 0], cy + radius * v[:, 1], z0 + radius + radius * v[:, 2]]))


def ellipsoid(rng, cx, cy, z0, rx=2.0, rz=3.1, n=3000):
    v = rng.normal(size=(n, 3))
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    return _noise(rng, np.column_stack([cx + rx * v[:, 0], cy + rx * v[:, 1], z0 + rz + rz * v[:, 2]]))


def cone(rng, cx, cy, z0, radius=2.4, height=6.2, n=3000):
    h = 1.0 - np.sqrt(rng.uniform(0.0, 1.0, n))          # uniform on the lateral surface
    a = rng.uniform(0.0, 2.0 * np.pi, n)
    r = radius * (1.0 - h)
    return _noise(rng, np.column_stack([cx + r * np.cos(a), cy + r * np.sin(a), z0 + height * h]))


def umbrella(rng, cx, cy, z0, rx=3.0, rz=1.5, n=3000):
    """Flattened ellipsoid (spreading dome): much wider than tall."""
    return ellipsoid(rng, cx, cy, z0, rx=rx, rz=rz, n=n)


def column(rng, cx, cy, z0, radius=0.8, height=6.0, n=3000):
    a = rng.uniform(0.0, 2.0 * np.pi, n)
    z = rng.uniform(0.0, height, n)
    return _noise(rng, np.column_stack([cx + radius * np.cos(a), cy + radius * np.sin(a), z0 + z]))


@pytest.mark.parametrize(("maker", "expected", "diameter"), [
    (sphere, "sphere", 4.2), (ellipsoid, "ellipsoid", 4.0), (cone, "cone", 4.8),
    (umbrella, "umbrella", 6.0), (column, "cylinder", 1.6),
])
def test_crown_shape_classes(maker, expected, diameter) -> None:
    rng = np.random.default_rng(11)
    pts = maker(rng, 5.0, 7.0, 2.5)
    fit = fit_crown_shape(pts, reference_xy=(5.0, 7.0))
    assert fit.shape == expected, (fit.shape, fit.residual)
    assert fit.status == "measured"
    assert fit.diameter_m == pytest.approx(diameter, abs=0.25)
    assert fit.base_z == pytest.approx(2.5, abs=0.8)  # narrow bases (umbrella) start a little higher
    assert np.hypot(*fit.offset_xy) < 0.2


def test_crown_offset_and_too_few_points() -> None:
    rng = np.random.default_rng(12)
    pts = ellipsoid(rng, 6.0, 7.0, 2.5)
    fit = fit_crown_shape(pts, reference_xy=(5.0, 7.0))
    assert fit.offset_xy[0] == pytest.approx(1.0, abs=0.15)
    few = fit_crown_shape(pts[:100])
    assert few.shape == "ellipsoid" and few.status == "assumed" and few.diameter_m is None


# ---- detect_trees end to end ---------------------------------------------------------------------------------------

def _tree_points(rng, crown_pts, cx, cy, *, trunk_d=0.35, trunk_top=2.6, n_trunk=1200, trunk_class=TREE_TRUNK):
    ground = float(_dtm(np.array([[cx, cy]]))[0])
    crown = crown_pts.copy()
    crown[:, 2] += ground
    chunks, classes = [crown], [np.full(len(crown), TREE)]
    if n_trunk:
        a = rng.uniform(0.0, 2.0 * np.pi, n_trunk)
        z = rng.uniform(0.0, trunk_top, n_trunk)
        t = np.column_stack([cx + trunk_d / 2 * np.cos(a), cy + trunk_d / 2 * np.sin(a), ground + z])
        chunks.append(t + rng.normal(0.0, 0.012, t.shape))
        classes.append(np.full(n_trunk, trunk_class))
    return np.vstack(chunks), np.concatenate(classes)


def _pointset(parts) -> PointSet:
    return PointSet(xyz=np.vstack([p[0] for p in parts]), classes=np.concatenate([p[1] for p in parts]).astype(np.uint16))


def test_detect_trees_shape_and_measured_trunk() -> None:
    rng = np.random.default_rng(21)
    parts = [
        _tree_points(rng, ellipsoid(rng, 0.0, 0.0, 2.5), 0.0, 0.0, trunk_d=0.35),
        _tree_points(rng, cone(rng, 14.0, 0.0, 2.0, radius=2.2, height=6.5), 14.0, 0.0, trunk_d=0.35, n_trunk=0),
    ]
    trees = detect_trees(_pointset(parts), _dtm)
    assert len(trees) == 2
    by_x = sorted(trees, key=lambda t: t.trunk_xy[0])
    rounded, conifer = by_x
    assert rounded.crown_shape in ("ellipsoid", "sphere") and rounded.crown_shape_status == "measured"
    assert conifer.crown_shape == "cone"
    assert rounded.trunk_measured and rounded.trunk_diameter_m == pytest.approx(0.35, abs=0.03)
    assert rounded.crown_base_z is not None and rounded.crown_top_z > rounded.crown_base_z
    # no trunk points: nothing is invented
    assert conifer.trunk_diameter_m is None and conifer.trunk_measured is False


def test_trunk_taken_from_tree_points_below_the_crown() -> None:
    rng = np.random.default_rng(22)
    part = _tree_points(rng, sphere(rng, 0.0, 0.0, 2.6), 0.0, 0.0, trunk_d=0.40, trunk_class=TREE)
    (tree,) = detect_trees(_pointset([part]), _dtm)
    assert tree.trunk_measured and tree.trunk_diameter_m == pytest.approx(0.40, abs=0.03)


def test_noisy_trunk_is_not_measured() -> None:
    rng = np.random.default_rng(23)
    crown = ellipsoid(rng, 0.0, 0.0, 2.5)
    ground = float(_dtm(np.array([[0.0, 0.0]]))[0])
    crown[:, 2] += ground
    a = rng.uniform(0.0, 2.0 * np.pi, 800)
    z = rng.uniform(0.0, 2.6, 800)
    t = np.column_stack([0.17 * np.cos(a), 0.17 * np.sin(a), ground + z]) + rng.normal(0.0, 0.08, (800, 3))
    ps = _pointset([(np.vstack([crown, t]), np.concatenate([np.full(len(crown), TREE), np.full(800, TREE_TRUNK)]))])
    (tree,) = detect_trees(ps, _dtm)
    assert tree.trunk_diameter_m is None and tree.trunk_measured is False


# ---- old pickles ---------------------------------------------------------------------------------------------------

def test_old_tree_instance_pickle_still_loads() -> None:
    tree = TreeInstance(3, (1.0, 2.0), 0.5, 7.0, 4.0, 0.3, np.arange(400), Evidence(2, 400, None), "ok", "")
    for name in ("crown_shape", "crown_shape_status", "crown_base_z", "crown_top_z", "crown_offset_xy",
                 "trunk_measured", "crown_fit_residual"):
        tree.__dict__.pop(name, None)          # what a pickle written before the new fields contains
    loaded = pickle.loads(pickle.dumps(tree))
    assert loaded.crown_shape == "ellipsoid" and loaded.crown_shape_status == "assumed"
    assert loaded.crown_base_z is None and loaded.trunk_measured is False and loaded.crown_offset_xy == (0.0, 0.0)
    (element,), _ = veg.build_vegetation_elements([loaded], [])
    assert element.params.crown_shape == "ellipsoid"


# ---- IFC -----------------------------------------------------------------------------------------------------------

def _tree_instance(tid=1, xy=(3.0, 4.0), shape="cone", trunk=0.35, base=1.03, crown_base=3.2, height=8.4) -> TreeInstance:
    return TreeInstance(tid, xy, base, height, 5.0, trunk, np.arange(400), Evidence(5, 400, 0.02), "ok", "",
                        crown_shape=shape, crown_shape_status="measured", crown_base_z=crown_base,
                        crown_top_z=base + height, trunk_measured=trunk is not None)


def _ifc(tmp_path: Path, instances):
    import ifcopenshell
    result = veg.build_vegetation(instances, [], [], _dtm)
    scene = Scene()
    for element in result.elements:
        scene.add(element)
    path = tmp_path / "trees.ifc"
    write_ifc(scene, path)
    return ifcopenshell.open(str(path))


def _tree_items(ifc):
    """(trunk, crown) face sets of the single Body representation of the first tree."""
    product = ifc.by_type("IfcGeographicElement")[0]
    reps = [r for r in product.Representation.Representations if r.RepresentationIdentifier == "Body"]
    assert len(reps) == 1 and reps[0].RepresentationType == "Tessellation"
    trunk, crown = reps[0].Items
    return trunk, crown


SHAPES = ["sphere", "ellipsoid", "cone", "umbrella", "cylinder"]


@pytest.mark.parametrize("shape", SHAPES)
def test_ifc_one_body_trunk_reaches_into_centred_crown(tmp_path: Path, shape: str) -> None:
    ifc = _ifc(tmp_path, [_tree_instance(shape=shape)])
    trunk, crown = _tree_items(ifc)
    assert len(crown.CoordIndex) <= 120 and 60 <= len(trunk.CoordIndex) + len(crown.CoordIndex) <= 120
    tz = [pt[2] for pt in trunk.Coordinates.CoordList]
    cz = [pt[2] for pt in crown.Coordinates.CoordList]
    crown_height = 1.03 + 8.4 - 3.2
    assert min(tz) == pytest.approx(1.03, abs=0.01)                        # trunk stands on the terrain
    assert max(tz) == pytest.approx(3.2 + 0.15 * crown_height, abs=0.01)   # ... and reaches 15 % into the crown
    assert min(cz) <= max(tz) and min(cz) == pytest.approx(3.2, abs=0.01)  # meshes overlap: connected
    assert max(cz) == pytest.approx(1.03 + 8.4, abs=0.01)
    cxy = np.mean([pt[:2] for pt in crown.Coordinates.CoordList], axis=0)
    assert cxy == pytest.approx([3.0, 4.0], abs=0.01)                      # crown centred on the trunk axis
    if shape == "cone":
        apex = max(crown.Coordinates.CoordList, key=lambda pt: pt[2])
        assert apex[2] == pytest.approx(max(cz)) and apex[:2] == pytest.approx((3.0, 4.0), abs=0.01)


def test_crown_never_starts_below_30_cm_above_terrain(tmp_path: Path) -> None:
    ifc = _ifc(tmp_path, [_tree_instance(crown_base=1.1)])                 # terrain 1.03 -> base clamped to 1.33
    _, crown = _tree_items(ifc)
    assert min(pt[2] for pt in crown.Coordinates.CoordList) == pytest.approx(1.03 + 0.3, abs=0.01)


def test_off_centre_trunk_under_crown_dxf_and_ifc(tmp_path: Path) -> None:
    inst = _tree_instance()
    inst.crown_offset_xy = (0.6, -0.4)                 # crown centre minus trunk position
    (element,), _ = veg.build_vegetation_elements([inst], [])
    assert element.params.trunk_xy == pytest.approx((3.0, 4.0)) and element.params.crown_offset == pytest.approx((0.6, -0.4))
    assert not any("clamped" in n for n in element.notes)
    crown_insert, stem = veg.to_dxf_entities([element], FRAME)["S-TREE"]
    assert crown_insert["name"] == "TREE" and crown_insert["point"][:2] == pytest.approx([2_683_003.6, 1_248_003.6])
    assert stem["name"] == "TREE_STEM" and stem["point"][:2] == pytest.approx([2_683_003.0, 1_248_004.0])
    ifc = _ifc(tmp_path, [inst])
    trunk, crown = _tree_items(ifc)
    assert np.mean([pt[:2] for pt in crown.Coordinates.CoordList], axis=0) == pytest.approx([3.6, 3.6], abs=0.01)
    assert np.mean([pt[:2] for pt in trunk.Coordinates.CoordList], axis=0) == pytest.approx([3.0, 4.0], abs=0.01)


def test_trunk_outside_crown_footprint_is_clamped_and_flagged() -> None:
    inst = _tree_instance()
    inst.crown_offset_xy = (3.0, 0.0)                  # crown radius 2.5, trunk radius 0.175: reach 2.325
    (element,), _ = veg.build_vegetation_elements([inst], [])
    p = element.params
    dist = float(np.hypot(*p.crown_offset))
    assert dist == pytest.approx(2.5 - 0.175, abs=0.01)
    assert (p.trunk_xy[0] + p.crown_offset[0], p.trunk_xy[1] + p.crown_offset[1]) == pytest.approx((6.0, 4.0))  # crown stays
    assert any("clamped" in n for n in element.notes)


def test_detect_trees_crown_at_points_trunk_measured_or_in_the_middle() -> None:
    rng = np.random.default_rng(41)
    crown = ellipsoid(rng, 6.0, 1.0, 2.5)              # lop-sided: points centred on x = 6
    # unmeasured: no trunk points -> the trunk stands at the crown centre
    (t0,) = detect_trees(_pointset([_tree_points(rng, crown, 5.0, 1.0, n_trunk=0)]), _dtm)
    assert t0.trunk_xy == pytest.approx((6.0, 1.0), abs=0.15) and np.hypot(*t0.crown_offset_xy) < 0.15
    # measured: trunk 0.5 m beside the crown centre keeps its own position, the crown stays on its points
    part = _tree_points(rng, crown, 5.5, 1.0, trunk_d=0.35)
    (t1,) = detect_trees(_pointset([part]), _dtm)
    assert t1.trunk_measured and t1.trunk_xy == pytest.approx((5.5, 1.0), abs=0.05)
    assert t1.crown_offset_xy == pytest.approx((0.5, 0.0), abs=0.15)


def test_rebase_trees_on_tin_surface() -> None:
    from types import SimpleNamespace
    gx, gy = np.meshgrid(np.linspace(-5.0, 15.0, 6), np.linspace(-5.0, 15.0, 6))
    vertices = np.column_stack([gx.ravel(), gy.ravel(), 0.5 + 0.1 * gx.ravel()])
    tri = []
    for r in range(5):
        for c in range(5):
            i = r * 6 + c
            tri += [(i, i + 1, i + 6), (i + 1, i + 7, i + 6)]
    tin = SimpleNamespace(vertices=vertices, triangles=np.array(tri))
    (element,), _ = veg.build_vegetation_elements([_tree_instance(xy=(3.0, 4.0))], [])
    top = element.params.base_z + element.params.height
    (moved,) = veg.rebase_trees_on_tin([element], tin)
    assert moved.params.base_z == pytest.approx(0.8, abs=1e-6)             # 0.5 + 0.1 * 3.0
    assert moved.params.base_z + moved.params.height == pytest.approx(top)  # the top of the tree stays


def test_no_inverted_cone_anywhere() -> None:
    import samcloud.bim.core.ifc.shapes as shapes_mod
    import samcloud.environment.vegetation as env_mod
    assert "inverted_cone" not in env_mod.CROWN_SHAPES and "inverted_cone" not in shapes_mod._CROWN_PROFILES
    assert "umbrella" in shapes_mod._CROWN_PROFILES


def test_ifc_geometry_spans_crown_and_full_height(tmp_path: Path) -> None:
    import ifcopenshell.geom
    ifc = _ifc(tmp_path, [_tree_instance(shape="cone")])
    product = ifc.by_type("IfcGeographicElement")[0]
    verts = np.asarray(ifcopenshell.geom.create_shape(ifcopenshell.geom.settings(), product).geometry.verts).reshape(-1, 3)
    assert np.ptp(verts[:, 0]) == pytest.approx(5.0, abs=0.05) and np.ptp(verts[:, 1]) == pytest.approx(5.0, abs=0.05)
    assert verts[:, 2].min() == pytest.approx(1.03, abs=0.01) and verts[:, 2].max() == pytest.approx(1.03 + 8.4, abs=0.01)


def _pset(ifc, name="SAMcloud_Tree") -> dict:
    product = ifc.by_type("IfcGeographicElement")[0]
    pset = next(p for p in product.IsDefinedBy if p.RelatingPropertyDefinition.Name == name)
    return {i.Name: i.NominalValue.wrappedValue for i in pset.RelatingPropertyDefinition.HasProperties}


def test_ifc_pset_measured_and_not_measured_trunk(tmp_path: Path) -> None:
    measured = _pset(_ifc(tmp_path, [_tree_instance(shape="cone")]))
    assert "CrownShape" not in measured and measured["Stammdurchmesser_m"] == pytest.approx(0.35)
    assert measured["StammGemessen"] is True
    assert measured["Hoehe_m"] == pytest.approx(8.4) and measured["Kronendurchmesser_m"] == pytest.approx(5.0)
    assert not {"Height", "CrownDiameter", "TrunkDiameter"} & set(measured)
    sub = tmp_path / "b"
    sub.mkdir()
    unmeasured = _pset(_ifc(sub, [_tree_instance(shape="cylinder", trunk=None)]))
    assert unmeasured["Stammdurchmesser_m"] == "nicht gemessen" and unmeasured["StammGemessen"] is False
    assert _pset(_ifc(sub, [_tree_instance(shape="cylinder", trunk=None)]), "SAMcloud_Herkunft")["Herkunft_trunk_diameter"] == "assumed"


def test_tree_element_params_carry_the_fit() -> None:
    (element,), _ = veg.build_vegetation_elements([_tree_instance(shape="cone")], [])
    assert element.params.crown_shape == "cone" and element.params.trunk_measured is True
    assert element.param_origin["crown_shape"] is ParamOrigin.MEASURED
    assert isinstance(element, TreeElement) and isinstance(element.params, TreeParams)


# ---- DXF -----------------------------------------------------------------------------------------------------------

def _doc(layers, tmp_path: Path):
    result = DxfPlanResult("p", "site", "outdoor", None, 1.0, Frame(np.zeros(3)), "t", layers, {},
                           dict.fromkeys(layers, 3), BackgroundImage(None, None, (0.0, 0.0, 10.0, 10.0)))
    path = tmp_path / "t.dxf"
    build_dxf_document(result).saveas(path)
    return ezdxf.readfile(path)


def test_dxf_tree_visible_attributes_and_trunk_only_when_measured(tmp_path: Path) -> None:
    elements, _ = veg.build_vegetation_elements(
        [_tree_instance(1, (3.0, 4.0), "cone", 0.35), _tree_instance(2, (8.0, 4.0), "ellipsoid", None)], [])
    doc = _doc(veg.to_dxf_entities(elements, FRAME), tmp_path)
    inserts = sorted((e for e in doc.modelspace() if e.dxftype() == "INSERT" and e.dxf.name == "TREE"), key=lambda e: e.dxf.insert.x)
    first, second = ({a.dxf.tag: a for a in e.attribs} for e in inserts)
    for tag in ("KRONE", "STAMM", "HOEHE"):
        assert not first[tag].is_invisible and not second[tag].is_invisible
        assert first[tag].dxf.height == pytest.approx(0.2)
    assert first["KRONE"].dxf.text == "KRONE=5.00 m" and first["HOEHE"].dxf.text == "HOEHE=8.40 m"
    assert "FORM" not in first and "FORM" not in second                          # the shape lives only in the IFC
    assert first["STAMM"].dxf.text == "STAMM=0.35 m" and second["STAMM"].dxf.text == "STAMM=nicht gemessen"
    assert first["HEIGHT"].is_invisible                                      # data attributes stay invisible
    stems = [e for e in doc.modelspace() if e.dxftype() == "INSERT" and e.dxf.name == "TREE_STEM"]
    assert len(stems) == 1 and stems[0].dxf.xscale == pytest.approx(0.175)   # trunk circle to scale, only for tree 1
    assert stems[0].dxf.insert.x == pytest.approx(2_683_003.0)


def test_dxf_manhole_visible_cover_height_two_decimals(tmp_path: Path) -> None:
    manhole = outdoor_symbol_insert("manhole", (3.0, 4.0, 431.2567), {"Z_DECKEL": 431.2567, "TYP": "Schacht"}, crown_radius=0.45)
    doc = _doc({"S-MANH": [manhole]}, tmp_path)
    (insert,) = [e for e in doc.modelspace() if e.dxftype() == "INSERT"]
    attribs = {a.dxf.tag: a for a in insert.attribs}
    assert attribs["Z_DECKEL"].dxf.text == "431.26"
    assert len(attribs["Z_DECKEL"].dxf.text.split(".")[1]) == 2
    assert not attribs["Z_DECKEL"].is_invisible and attribs["TYP"].is_invisible
    assert attribs["Z_DECKEL"].dxf.height == pytest.approx(0.2)
    assert insert.dxf.xscale == pytest.approx(0.45 / 0.3)                    # symbol scaled to the measured diameter


# ---- sparse / messy crowns (real clouds are partial, layered and contaminated) -------------------------------------

def _layered_conifer(rng, n=700, outlier_share=0.3):
    """Layered branches on a cone (radius shrinks with height), 3 cm noise, 30 % scattered inside-envelope outliers."""
    n_out = int(n * outlier_share)
    layers = np.arange(0.4, 6.0, 0.7)                      # branch whorls
    h = rng.choice(layers, n - n_out) + rng.normal(0.0, 0.05, n - n_out)
    h = np.clip(h, 0.0, 6.2)
    r = 2.6 * (1.0 - h / 6.4) * np.sqrt(rng.uniform(0.3, 1.0, n - n_out))
    a = rng.uniform(0.0, 2.0 * np.pi, n - n_out)
    pts = np.column_stack([r * np.cos(a), r * np.sin(a), 2.0 + h])
    ho = rng.uniform(0.0, 6.0, n_out)
    ro = 2.6 * (1.0 - ho / 6.4) * np.sqrt(rng.uniform(0.0, 1.0, n_out)) * 1.05
    ao = rng.uniform(0.0, 2.0 * np.pi, n_out)
    out = np.column_stack([ro * np.cos(ao), ro * np.sin(ao), 2.0 + ho])
    return _noise(rng, np.vstack([pts, out]))


def test_sparse_noisy_layered_conifer_is_a_cone() -> None:
    for seed in (31, 32, 33):
        fit = fit_crown_shape(_layered_conifer(np.random.default_rng(seed)))
        assert fit.shape == "cone", (seed, fit.shape, fit.residuals)


def test_sparse_hedge_box_is_not_a_cone() -> None:
    rng = np.random.default_rng(34)
    n = 600
    x = rng.uniform(-2.0, 2.0, n)
    y = rng.uniform(-0.5, 0.5, n)
    z = rng.uniform(0.0, 2.0, n)
    side = rng.random(n) < 0.35                            # a third of the points on the top, the rest anywhere
    z[side] = 2.0
    fit = fit_crown_shape(_noise(rng, np.column_stack([x, y, z + 0.1])))
    assert fit.shape != "cone", fit.residuals
