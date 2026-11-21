"""Curbs, gutter stones, fences and walls sit flush on the final TIN (synthetic stepped surface)."""
from __future__ import annotations

from types import SimpleNamespace

import ifcopenshell
import ifcopenshell.geom
import numpy as np
from shapely.geometry import LineString, box

from samcloud.bim.core.ifc import write_ifc
from samcloud.bim.core.model import Scene, TerrainElement, TerrainParams
from samcloud.drone import contracts as c
from samcloud.drone import qa_cad
from samcloud.dxf.las_source import Frame
from samcloud.dxf.result import BackgroundImage, DxfPlanResult
from samcloud.dxf.writer import build_dxf_document
from samcloud.outdoor_model import civil, site_walls
from samcloud.terrain.tin import tin_sampler

E0, N0, H0 = 2_600_000.0, 1_200_000.0, 430.0
FRAME = c.LocalFrame(origin_lv95=(E0, N0, H0))
CFG = c.load_drone_config()
TOP = 0.12
OUTLINE = LineString([(-1, 0), (21, 0)])  # the pavement outline: where the TIN has its step top edge


def _tin():
    """Road (z = 0, y < 0) and sidewalk (z = 0.12, y > 0) meeting in a vertical step at y = 0 (no triangle covers the step face)."""
    v = np.array([[-1, -6, 0], [21, -6, 0], [21, 0, 0], [-1, 0, 0], [-1, 0, TOP], [21, 0, TOP], [21, 6, TOP], [-1, 6, TOP]], float)
    t = np.array([[0, 1, 2], [0, 2, 3], [4, 5, 6], [4, 6, 7]])
    return SimpleNamespace(vertices=v, triangles=t, labels=np.zeros(4, dtype=np.uint8))


def _axis(rng, y=0.0, z_noise=0.012):
    x = np.linspace(0, 20, 9)
    pts = np.column_stack([x, np.full(9, y), TOP + rng.normal(0, z_noise, 9)])
    return c.CurbAxis(0, pts, c.Evidence(points=300), foot_z=pts[:, 2] - 0.30, road_side=1, gutter_width=0.30)  # wrong side + wrong foot on purpose


def test_surface_sampler_keeps_the_step():
    sample = tin_sampler(_tin())
    z = sample(np.array([[5.0, -0.1], [5.0, 0.1], [5.0, 9.0]]))
    assert abs(z[0]) < 1e-9 and abs(z[1] - TOP) < 1e-9 and np.isnan(z[2])


def test_curb_top_and_gutter_follow_the_surface(tmp_path):
    rng = np.random.default_rng(41)
    pavement = box(-1, -6, 21, 0).union(box(-1, 0, 21, 6))
    elements, _ = civil.build_civil_elements([_axis(rng)], config=CFG, surface=tin_sampler(_tin()), pavement=pavement, outline=OUTLINE)
    (curb,) = elements
    p = curb.params
    assert np.abs(np.asarray(p.axis)[:, 2] - TOP).max() < 1e-6          # top = upper-side surface, not the noisy points
    assert p.road_side == -1                                              # the lower (road) side is on -y of a +x axis
    assert np.abs(np.asarray(p.foot_z)).max() < 1e-6 and p.gutter_width == 0.30
    assert np.abs(np.asarray(p.gutter_axis)[:, 2]).max() < 1e-6           # gutter top = road surface
    scene = Scene()
    scene.settings.origin_offset = (E0, N0, H0)
    v, t = _tin().vertices.tolist(), _tin().triangles.tolist()
    scene.add(TerrainElement(params=TerrainParams(vertices=[tuple(a) for a in v], triangles=[tuple(a) for a in t])))
    scene.add(curb)
    path = tmp_path / "s.ifc"
    write_ifc(scene, path)
    f = ifcopenshell.open(str(path))
    settings = ifcopenshell.geom.settings()
    settings.set("use-world-coords", True)
    meshes = {x.ObjectType: np.asarray(ifcopenshell.geom.create_shape(settings, x).geometry.verts).reshape(-1, 3)
              for x in f.by_type("IfcCivilElement")}
    gutter, body = meshes["Wasserstein"], meshes["Randstein"]
    assert abs(gutter[:, 2].max()) < 0.005 and gutter[:, 1].max() < 0.0   # flush with the road, on the road side
    assert abs(body[:, 2].max() - TOP) < 0.005
    # QA rule 19 on the real writer output; then a gutter lifted 5 cm above the surface is reported
    entities = civil.to_dxf_entities(elements, FRAME, config=CFG)
    result = DxfPlanResult("t", "site", "outdoor", None, 1.0, Frame(np.zeros(3)), "t", entities, {}, dict.fromkeys(entities, 7),
                           BackgroundImage(None, None, (E0, N0, E0 + 20, N0 + 20)))
    dxf = tmp_path / "s.dxf"
    build_dxf_document(result).saveas(dxf)
    rule = {r["rule"]: r for r in qa_cad.count_consistency(civil.category_counts(elements), dxf, path)}["19_gutter_curb_flush_with_surface"]
    assert rule["ok"] and rule["value"]["gutters"] == 1 and rule["value"]["curbs"] == 1, rule
    lifted = elements[0].model_copy(update={"params": p.model_copy(update={"gutter_axis": [(x, y, z + 0.05) for x, y, z in p.gutter_axis]})})
    scene2 = Scene()
    scene2.settings.origin_offset = (E0, N0, H0)
    scene2.add(TerrainElement(params=TerrainParams(vertices=[tuple(a) for a in v], triangles=[tuple(a) for a in t])))
    scene2.add(lifted)
    path2 = tmp_path / "s2.ifc"
    write_ifc(scene2, path2)
    bad = {r["rule"]: r for r in qa_cad.count_consistency(civil.category_counts([lifted]), dxf, path2)}["19_gutter_curb_flush_with_surface"]
    assert not bad["ok"] and bad["offenders"]


def test_gutter_only_where_the_road_side_is_pavement():
    rng = np.random.default_rng(42)
    sampler = tin_sampler(_tin())
    only_sidewalk = box(-1, 0, 21, 6)  # the lower side is bare terrain: a curb against terrain gets no gutter
    (bare,), _ = civil.build_civil_elements([_axis(rng)], config=CFG, surface=sampler, pavement=only_sidewalk, outline=OUTLINE)
    assert bare.params.gutter_width == 0.0 and bare.params.gutter_axis is None and bare.evidence["gutter"] == "no pavement"
    near = box(-1, -6, 21, -0.1)  # road polygon 10 cm beside the curb face is within reach
    (paved,), _ = civil.build_civil_elements([_axis(rng)], config=CFG, surface=sampler, pavement=near, outline=OUTLINE)
    assert paved.params.gutter_width == 0.30


def test_flush_curb_has_no_gutter_and_nothing_above_the_surface():
    flat = SimpleNamespace(vertices=np.array([[-1, -6, 0], [21, -6, 0], [21, 6, 0], [-1, 6, 0]], float), triangles=np.array([[0, 1, 2], [0, 2, 3]]),
                           labels=np.zeros(2, dtype=np.uint8))
    rng = np.random.default_rng(43)
    axis = _axis(rng)
    (curb,), _ = civil.build_civil_elements([axis], config=CFG, surface=tin_sampler(flat), pavement=box(-1, -6, 21, 6), outline=OUTLINE)
    assert curb.params.gutter_width == 0.0 and curb.evidence["flush"] is True
    assert np.asarray(curb.params.axis)[:, 2].max() <= 0.005


def test_fence_and_wall_bottoms_sit_on_the_tin():
    rng = np.random.default_rng(44)
    tin = SimpleNamespace(vertices=np.array([[-2, -2, 0.0], [14, -2, 0.8], [14, 8, 0.8], [-2, 8, 0.0]]), triangles=np.array([[0, 1, 2], [0, 2, 3]]),
                          labels=np.zeros(2, dtype=np.uint8))
    sampler = tin_sampler(tin)
    t = rng.uniform(0, 12, 4000)
    xy = np.column_stack([t, np.full(4000, 3.0)]) + rng.normal(0, 0.01, (4000, 2))
    z = sampler(xy) + 1.2 * np.where(rng.random(4000) < 0.3, 1.0, rng.uniform(0, 1, 4000)) + rng.normal(0, 0.004, 4000)
    ps = c.PointSet(np.column_stack([xy, z]), np.full(4000, c.FENCE, dtype=np.uint16))
    cfg = {"site": {"fence": {**CFG["site"]["fence"], "terrain_radius_m": 0.0}, "wall": CFG["site"]["wall"]}}
    (fence,) = site_walls.fence_features(ps, sampler, config=cfg)
    assert np.abs(fence.bottom_z - sampler(fence.axis[:, :2])).max() < 0.005
    wall_ps = c.PointSet(np.column_stack([xy, z]), np.full(4000, c.SITE_WALL, dtype=np.uint16))
    wall_cfg = {"site": {"fence": CFG["site"]["fence"], "wall": {**CFG["site"]["wall"], "terrain_radius_m": 0.0}}}
    (wall,) = site_walls.site_wall_features(wall_ps, sampler, config=wall_cfg)
    assert np.abs(wall.base_z_axis - sampler(wall.axis)).max() < 0.005


def test_gutter_is_draped_on_a_sloping_road_and_needs_a_step_at_a_pavement_outline():
    # road falls 4 % across the strip (curb-side edge higher than the outer edge): the slab top follows it exactly
    v = np.array([[-1, -6, -0.24], [21, -6, -0.24], [21, 0, 0.0], [-1, 0, 0.0], [-1, 0, TOP], [21, 0, TOP], [21, 6, TOP], [-1, 6, TOP]], float)
    tin = SimpleNamespace(vertices=v, triangles=np.array([[0, 1, 2], [0, 2, 3], [4, 5, 6], [4, 6, 7]]), labels=np.zeros(4, dtype=np.uint8))
    surface = tin_sampler(tin)
    rng = np.random.default_rng(45)
    (curb,), _ = civil.build_civil_elements([_axis(rng)], config=CFG, surface=surface, pavement=box(-1, -6, 21, 0), outline=OUTLINE)
    p = curb.params
    gz_in, gz_out = np.asarray(p.gutter_axis)[:, 2], np.asarray(p.gutter_z_outer)
    xy = np.asarray(p.gutter_axis)[:, :2]
    assert np.abs(gz_in - surface(np.column_stack([xy[:, 0], np.full(len(xy), -0.06)]))).max() < 0.005     # curb-side edge on the TIN
    assert np.abs(gz_out - surface(np.column_stack([xy[:, 0], np.full(len(xy), -0.36)]))).max() < 0.005    # outer edge on the TIN
    assert (gz_in - gz_out).min() > 0.01                                                                    # draped, not a flat slab
    # a curb far from every pavement outline has no step in the TIN: no gutter, top = the plain surface under it
    far = LineString([(-1, 4.0), (21, 4.0)])
    (loose,), _ = civil.build_civil_elements([_axis(rng, y=-3.0)], config=CFG, surface=surface, pavement=box(-1, -6, 21, 0), outline=far)
    assert loose.params.gutter_width == 0.0 and loose.evidence["gutter"] in ("no step", "flush")
    assert np.abs(np.asarray(loose.params.axis)[:, 2] - surface(np.asarray(loose.params.axis)[:, :2])).max() < 1e-6


def test_curb_axis_within_snap_distance_lies_on_the_outline():
    rng = np.random.default_rng(46)
    tin = _tin()
    (curb,), _ = civil.build_civil_elements([_axis(rng, y=0.10)], config=CFG, surface=tin_sampler(tin), pavement=box(-1, -6, 21, 0), outline=OUTLINE)
    assert np.abs(np.asarray(curb.params.axis)[:, 1]).max() < 1e-9              # snapped onto the step line
    assert np.abs(np.asarray(curb.params.axis)[:, 2] - TOP).max() < 1e-6        # so its top is the upper side of the step


def _ramp_tin():
    """Road with a calm part (x < 12) and a ramp part (x > 12) whose surface falls 0.3 m within 0.4 m of the curb (across the gutter strip)."""
    v = np.array([[-1, -6, 0], [12, -6, 0], [12, 0, 0], [-1, 0, 0],                                   # calm road, z = 0
                  [12, -0.4, 0], [21, -0.4, -0.3], [21, 0, 0], [12, 0, 0],                             # ramp strip beside the curb
                  [12, -6, 0], [21, -6, -0.3], [21, -0.4, -0.3], [12, -0.4, 0],                        # road below the ramp
                  [-1, 0, TOP], [21, 0, TOP], [21, 6, TOP], [-1, 6, TOP]], float)
    t = np.array([[0, 1, 2], [0, 2, 3], [4, 5, 6], [4, 6, 7], [8, 9, 10], [8, 10, 11], [12, 13, 14], [12, 14, 15]])
    return SimpleNamespace(vertices=v, triangles=t, labels=np.zeros(len(t), dtype=np.uint8))


def test_gutter_is_dropped_where_the_surface_is_a_ramp_not_a_road(tmp_path):
    """Regression (Wasserstein 5): on a ramp / stair-like surface no draped slab can match the TIN, so only the calm stretch gets a gutter."""
    rng = np.random.default_rng(47)
    tin = _ramp_tin()
    surface = tin_sampler(tin)
    (curb,), _ = civil.build_civil_elements([_axis(rng)], config=CFG, surface=surface, pavement=box(-1, -6, 21, 0), outline=OUTLINE)
    p = curb.params
    assert p.gutter_width > 0 and p.gutter_axis is not None
    ga = np.asarray(p.gutter_axis)
    assert 11.0 < ga[:, 0].max() <= 14.5 and ga[:, 0].min() < 0.5      # the calm stretch only (the strip falls > 5 cm across from about x = 14), not the whole curb
    # IFC + QA: every gutter face centre lies on the surface the QA samples
    scene = Scene()
    scene.settings.origin_offset = (E0, N0, H0)
    scene.add(TerrainElement(params=TerrainParams(vertices=[tuple(a) for a in tin.vertices.tolist()], triangles=[tuple(a) for a in tin.triangles.tolist()])))
    scene.add(curb)
    path = tmp_path / "ramp.ifc"
    write_ifc(scene, path)
    entities = civil.to_dxf_entities([curb], FRAME, config=CFG)
    result = DxfPlanResult("t", "site", "outdoor", None, 1.0, Frame(np.zeros(3)), "t", entities, {}, dict.fromkeys(entities, 7),
                           BackgroundImage(None, None, (E0, N0, E0 + 20, N0 + 20)))
    dxf = tmp_path / "ramp.dxf"
    build_dxf_document(result).saveas(dxf)
    rule = {r["rule"]: r for r in qa_cad.count_consistency(civil.category_counts([curb]), dxf, path)}["19_gutter_curb_flush_with_surface"]
    assert rule["ok"] and rule["value"]["max_median_gutter_m"] < 0.005, rule
    # a curb that runs only along the ramp gets no gutter at all
    ramp_axis = c.CurbAxis(1, np.array([[16.0, 0.0, TOP], [20.0, 0.0, TOP]]), c.Evidence(points=100), foot_z=np.array([0.0, -0.3]), road_side=1, gutter_width=0.30)
    (bare,), _ = civil.build_civil_elements([ramp_axis], config=CFG, surface=surface, pavement=box(-1, -6, 21, 0), outline=OUTLINE)
    assert bare.params.gutter_width == 0.0 and bare.evidence["gutter"] == "no calm surface"
