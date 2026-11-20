"""Tests for samcloud.outdoor_model.terrain (terrain element, TIN/contour/spot DXF, contour interval)."""

from __future__ import annotations

import ezdxf
import numpy as np
import pytest
from shapely.geometry import Polygon, box

from samcloud.bim.core.ifc import write_ifc
from samcloud.bim.core.model import Scene, TerrainElement, TerrainParams
from samcloud.drone import contracts as c
from samcloud.drone import qa_cad
from samcloud.dxf.las_source import Frame
from samcloud.dxf.result import BackgroundImage, DxfPlanResult
from samcloud.dxf.writer import build_dxf_document
from samcloud.outdoor_model import pavement as pv
from samcloud.outdoor_model import terrain as tr

pytest.importorskip("samcloud.terrain.tin")
from samcloud.terrain.tin import build_labeled_tin  # noqa: E402

E0, N0, H0 = 2_600_000.0, 1_200_000.0, 430.0
FRAME = c.LocalFrame(origin_lv95=(E0, N0, H0))
CFG = c.load_drone_config()
RES = 0.25
BOUNDS = (0.0, 0.0, 30.0, 20.0)


def _scene(slope: float = 0.02, seed: int = 3, spacing: float = 2.0):
    road = Polygon([(0, 0), (30, 0), (30, 9.93), (15, 9.93), (15, 10.07), (0, 10.07)])
    polys = [c.PavementPolygon(0, "road", road), c.PavementPolygon(1, "sidewalk", box(0, 10.0, 30, 14)),
             c.PavementPolygon(2, "path", box(5, 16, 25, 19))]
    lawns = [box(0, 14, 30, 20)]
    curbs = [c.CurbAxis(0, np.array([[0.0, 10.0, 0.10], [30.0, 10.0, 0.10 + 30 * slope]]))]
    rng = np.random.default_rng(seed)
    gx, gy = np.meshgrid(np.arange(-1, 31.5, spacing), np.arange(-1, 21.5, spacing))
    xy = np.column_stack([gx.ravel(), gy.ravel()]) + rng.normal(0, 0.1, (gx.size, 2))
    z = slope * xy[:, 0] + 0.01 * slope * xy[:, 1] + rng.normal(0, 0.004, len(xy))
    ground = np.column_stack([xy, z])
    topo = pv.pavement_topology(polys, lawns, curbs, config=CFG)
    tin = build_labeled_tin(ground, regions=topo.regions(), breaklines=[curbs[0].points])
    return polys, lawns, curbs, tin


def _dtm(slope: float, noise: float, seed: int = 4, measured_to_x: float = 24.0):
    ny, nx = int(20 / RES), int(30 / RES)
    xs = (np.arange(nx) + 0.5) * RES
    ys = 20.0 - (np.arange(ny) + 0.5) * RES  # north-up
    gx, gy = np.meshgrid(xs, ys)
    z = slope * gx + 0.01 * slope * gy + np.random.default_rng(seed).normal(0, noise, gx.shape)
    validity = np.where(gx < measured_to_x, 1, 2).astype(np.uint8)
    return z.astype(np.float32), validity


def _write_dxf(layers: dict, path) -> None:
    result = DxfPlanResult("t", "site", "outdoor", None, 1.0, Frame(np.zeros(3)), "t", layers, {},
                           dict.fromkeys(layers, 7), BackgroundImage(None, None, (E0, N0, E0 + 10, N0 + 10)))
    build_dxf_document(result).saveas(path)


@pytest.fixture(scope="module")
def scene():
    polys, lawns, curbs, tin = _scene()
    pave, _ = pv.build_pavement_elements(polys, lawns, curbs, tin, config=CFG)
    terrain, qa = tr.build_terrain_elements(tin, config=CFG)
    return polys, lawns, curbs, tin, pave, terrain, qa


def _rule10(path):
    return [r for r in qa_cad.check_ifc(path, CFG) if r["rule"].startswith("10_")][0]


def test_terrain_has_no_triangle_inside_pavement_in_written_ifc(scene, tmp_path) -> None:
    _p, _l, _c, tin, pave, terrain, qa = scene
    assert len(terrain) == 1 and isinstance(terrain[0], TerrainElement)
    assert not [q for q in qa if q.category == "terrain" and "no non-pavement" in q.reason]
    sc = Scene()
    for e in (*pave, *terrain):
        sc.add(e)
    path = tmp_path / "scene.ifc"
    write_ifc(sc, path)
    rule = _rule10(path)
    assert rule["ok"], rule
    assert rule["value"]["triangles"] > 100 and rule["value"]["inside_pavement"] == 0
    # negative control: the full TIN as terrain overlaps the pavement and is caught by the rule
    full = TerrainElement(id="full", params=TerrainParams(vertices=[tuple(v) for v in tin.vertices.tolist()],
                                                          triangles=[tuple(t) for t in tin.triangles.tolist()]))
    sc2 = Scene()
    for e in pave:
        sc2.add(e)
    sc2.add(full)
    path2 = tmp_path / "full.ifc"
    write_ifc(sc2, path2)
    assert not _rule10(path2)["ok"]


def test_terrain_drops_pavement_labels_keeps_lawn(scene) -> None:
    _p, _l, _c, tin, _pave, terrain, _qa = scene
    t = terrain[0]
    labels = tin.labels
    assert (labels == c.TIN_LABELS["lawn"]).any() and (labels == c.TIN_LABELS["road"]).any()
    assert t.evidence["lawn_triangles"] == int((labels == c.TIN_LABELS["lawn"]).sum()) or t.evidence["dropped_degenerate"] > 0
    expected = int(np.isin(labels, [c.TIN_LABELS["terrain"], c.TIN_LABELS["lawn"]]).sum())
    assert len(t.params.triangles) == expected - t.evidence["dropped_degenerate"]
    assert t.evidence["pavement_triangles_removed"] == int(np.isin(labels, [1, 2, 3]).sum())
    centroids = np.asarray(t.params.vertices)[np.asarray(t.params.triangles)].mean(axis=1)
    for kind in ("road", "sidewalk", "path"):
        poly = Polygon()
        from shapely.ops import unary_union
        poly = unary_union([p.polygon for p in pv.pavement_topology(*_scene()[:3], config=CFG).polygons if p.kind == kind])
        from shapely import contains_xy
        assert not contains_xy(poly, centroids[:, 0], centroids[:, 1]).any()
    assert t.param_origin["vertices"].value == "fitted"


def test_degenerate_triangles_are_dropped_with_qa_entry() -> None:
    v = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0.01], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0], [2.5, 1e-9, 0.0]])
    tin = c.LabeledTin(vertices=v, triangles=np.array([[0, 1, 2], [1, 3, 2], [4, 5, 6]]), labels=np.zeros(3, dtype=np.uint8))
    elements, qa = tr.build_terrain_elements(tin, config=CFG)
    assert len(elements[0].params.triangles) == 2
    assert qa and qa[0].values["dropped"] == 1
    empty, qa2 = tr.build_terrain_elements(c.LabeledTin(v, np.array([[0, 1, 2]]), np.array([1], dtype=np.uint8)), config=CFG)
    assert empty == [] and "no non-pavement" in qa2[0].reason


@pytest.mark.parametrize("slope,noise,expected", [(0.05, 0.01, 0.2), (0.05, 0.08, 0.5), (0.25, 0.005, 0.5), (0.0, 0.002, 0.2)])
def test_contour_minor_interval_auto_choice_is_deterministic(slope, noise, expected) -> None:
    dtm, validity = _dtm(slope, noise)
    first = tr.choose_minor_interval(dtm, validity, RES, config=CFG)
    second = tr.choose_minor_interval(dtm.copy(), validity.copy(), RES, config=CFG)
    assert first == second
    assert first[0] == expected and first[1]["mode"] == "auto"
    assert first[1]["noise_rms_m"] is not None and "reason" in first[1]


def test_contour_interval_ignores_interpolated_cells_and_honours_fixed_value() -> None:
    dtm, validity = _dtm(0.05, 0.005)
    noisy = dtm.copy()
    noisy[validity == 2] += np.random.default_rng(1).normal(0, 0.5, int((validity == 2).sum())).astype(np.float32)
    assert tr.choose_minor_interval(noisy, validity, RES, config=CFG)[0] == 0.2
    fixed = {**CFG, "dxf": {**CFG["dxf"], "contour_minor_m": 0.25}}
    interval, info = tr.choose_minor_interval(noisy, validity, RES, config=fixed)
    assert interval == 0.25 and info["mode"] == "configured"
    assert tr.choose_minor_interval(np.full((3, 3), np.nan), np.zeros((3, 3), np.uint8), RES, config=CFG)[0] == 0.5


def test_dxf_tin_contours_spots_pass_qa_rules_3_4_6(scene, tmp_path) -> None:
    _p, _l, curbs, tin, _pave, terrain, _qa = scene
    dtm, validity = _dtm(0.05, 0.01)
    entities, info = tr.terrain_dxf_entities(
        terrain, FRAME, tin=tin, dtm=dtm, validity=validity, dtm_bounds=BOUNDS, resolution=RES,
        curbs=curbs, manholes=[(12.3, 4.2)], config=CFG)
    assert info["minor_interval_m"] == 0.2 and info["contour_count"] > 0
    assert len(entities["S-TOPO-TIN"]) == len(tin.triangles)
    assert {"S-TOPO-MAJR", "S-TOPO-MINR", "S-ELEV"} <= set(entities)
    assert all(e["type"] == "spline" for layer in ("S-TOPO-MAJR", "S-TOPO-MINR") for e in entities[layer])
    # contours only on the measured cells (x < 24 m)
    xs = [p[0] - E0 for layer in ("S-TOPO-MAJR", "S-TOPO-MINR") for e in entities[layer] for p in e["fit_points"]]
    assert max(xs) < 24.5
    levels = {round(e["elevation"], 3) for e in entities["S-TOPO-MAJR"]}
    assert levels == {431.0}
    path = tmp_path / "terrain.dxf"
    _write_dxf(entities, path)
    results = qa_cad.check_dxf(path, CFG)
    for prefix in ("3_", "4_", "6_"):
        assert all(r["ok"] for r in results if r["rule"].startswith(prefix)), (prefix, results)
    doc = ezdxf.readfile(str(path))
    assert doc.layers.get("S-TOPO-TIN").is_frozen() or not doc.layers.get("S-TOPO-TIN").is_on()
    assert len(doc.modelspace().query("3DFACE")) == len(tin.triangles)
    texts = [e.dxf.text for e in doc.modelspace().query("TEXT") if e.dxf.layer == "S-ELEV"]
    assert texts and all(t.startswith("+43") for t in texts)


def test_spot_heights_cover_manhole_curb_breakpoints_and_grid(scene) -> None:
    _p, _l, curbs, tin, _pave, terrain, _qa = scene
    entities = tr.to_dxf_entities(terrain, FRAME, tin=tin, curbs=curbs, manholes=[(12.3, 4.2)], config=CFG)
    spots = entities["S-ELEV"]
    texts = [e for e in spots if e["type"] == "text"]
    lines = [e for e in spots if e["type"] == "line"]
    assert len(lines) == 2 * len(texts) > 10
    xy = np.array([[t["point"][0] - E0 - 0.5, t["point"][1] - N0] for t in texts])
    assert np.hypot(xy[:, 0] - 12.3, xy[:, 1] - 4.2).min() < 0.5          # manhole
    assert np.hypot(xy[:, 0] - 0.0, xy[:, 1] - 10.0).min() < 0.5          # curb break point
    # spot height = TIN height + datum, formatted with sign and two decimals
    sampler = tr.TinSampler(tin.vertices, tin.triangles)
    z = sampler.sample(np.array([[12.3, 4.2]]))[0]
    near = min(texts, key=lambda t: abs(t["point"][0] - E0 - 0.5 - 12.3) + abs(t["point"][1] - N0 - 4.2))
    assert near["text"] == f"{z + H0:+.2f}"


def test_tin_sampler_is_exact_on_a_plane_and_nan_outside() -> None:
    rng = np.random.default_rng(2)
    xy = rng.uniform(0, 10, (200, 2))
    v = np.column_stack([xy, 0.3 * xy[:, 0] - 0.1 * xy[:, 1] + 2.0])
    from scipy.spatial import Delaunay
    sampler = tr.TinSampler(v, Delaunay(xy).simplices)
    q = np.array([[5.01, 5.02], [2.3, 7.7], [-5.0, 5.0]])
    z = sampler.sample(q)
    assert z[:2] == pytest.approx(0.3 * q[:2, 0] - 0.1 * q[:2, 1] + 2.0, abs=1e-9)
    assert np.isnan(z[2])


def test_evidence_reports_contour_choice_and_category_counts(scene) -> None:
    _p, _l, _c, tin, _pave, terrain, _qa = scene
    dtm, validity = _dtm(0.05, 0.08)
    elements, _ = tr.build_terrain_elements(tin, dtm=dtm, validity=validity, resolution=RES, config=CFG)
    ev = elements[0].evidence
    assert ev["contour_minor_m"] == 0.5 and ev["dtm_noise_rms_m"] > 0.05 and "contour_choice" in ev
    assert tr.category_counts(elements) == {"terrain": 1}
    assert tr.category_counts([]) == {"terrain": 0}
