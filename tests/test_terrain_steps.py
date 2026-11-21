"""Vertical steps (top + bottom edge, 2.5D) and manholes as TIN constraints."""

from __future__ import annotations

import numpy as np
from shapely.geometry import box

from samcloud.drone import qa_cad
from samcloud.drone.contracts import TIN_LABELS
from samcloud.terrain import steps as st
from samcloud.terrain.tin import build_labeled_tin, tin_sampler
from samcloud.terrain.flush import plan_flush
from types import SimpleNamespace
from samcloud.terrain.clean import tin_deviation

REGIONS = {"road": [box(0, 0, 10, 20)], "sidewalk": [box(10, 0, 20, 20)]}
CURB = SimpleNamespace(points=np.array([[10.0, 0.0, 0.12], [10.0, 10.0, 0.12], [10.0, 20.0, 0.12]]), foot_z=np.zeros(3), road_side=1)  # road on the left


def _ground(height, seed=3):
    rng = np.random.default_rng(seed)
    g = np.mgrid[0.125:20:0.25, 0.125:20:0.25].reshape(2, -1).T + rng.uniform(-0.04, 0.04, (6400, 2))
    return np.column_stack((g, height(g) + rng.normal(0, 0.002, len(g))))


def _sampler(height):
    return lambda xy: height(np.asarray(xy, float).reshape(-1, 2)) + 0.002 * np.sin(np.asarray(xy, float).reshape(-1, 2)[:, 0] * 3.1)


def _curb_height(xy):
    return np.where(xy[:, 0] >= 10.0, 0.12, 0.0)


def _areas(tin):
    v = tin.vertices
    a, b, c = (v[tin.triangles[:, i]] for i in range(3))
    area3 = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
    xy = 0.5 * ((b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0]))
    return area3, xy


def test_curb_step_has_top_and_bottom_edge_one_cm_apart() -> None:
    sampler = _sampler(_curb_height)
    runs = st.detect_steps(REGIONS, sampler, curbs=[CURB], config={})
    assert len(runs) == 1 and runs[0].source == "curb" and abs(runs[0].length - 20.0) < 0.1 and runs[0].lower_side == 1
    tin = build_labeled_tin(_ground(_curb_height), regions=REGIONS, dtm_sampler=sampler, steps=runs,
                            tolerance=0.02, initial_spacing=2.0, max_vertices=20000)
    report = tin.step_report
    assert report["segments"] >= 70 and abs(report["length_m"] - 20.0) < 0.1 and abs(report["mean_jump_m"] - 0.12) < 0.01
    v = tin.vertices
    top = v[(np.abs(v[:, 0] - 10.0) < 1e-6) & (v[:, 2] > 0.06)]
    bottom = v[(np.abs(v[:, 0] - 9.99) < 1e-6) & (v[:, 2] < 0.06)]
    assert len(top) >= 70 and len(bottom) >= 70
    assert np.all(np.abs(top[:, 2] - 0.12) < 0.005) and np.all(np.abs(bottom[:, 2]) < 0.005)
    assert sorted(top[1:-1, 1].round(6)) == sorted(bottom[:, 1].round(6)) or abs(len(top) - len(bottom)) <= 2
    area3, xy = _areas(tin)
    assert area3.min() > 1e-6 and np.abs(xy).min() > 1e-8
    assert (xy < 0).sum() in (0, len(xy))          # no flipped triangles
    # strips (one vertex on the top edge, one on the bottom edge) carry the label step
    on_top = np.isin(tin.triangles, np.flatnonzero((np.abs(v[:, 0] - 10.0) < 1e-6) & (v[:, 2] > 0.06))).any(axis=1)
    on_bottom = np.isin(tin.triangles, np.flatnonzero(np.abs(v[:, 0] - 9.99) < 1e-6)).any(axis=1)
    strips = on_top & on_bottom
    assert (tin.labels[strips] == TIN_LABELS["step"]).sum() >= 140   # strips carry their own label
    centres = v[tin.triangles].mean(axis=1)
    assert np.all(tin.labels[centres[:, 0] > 10.2] == TIN_LABELS["sidewalk"])
    assert np.all(tin.labels[centres[:, 0] < 9.8] == TIN_LABELS["road"])
    # still a valid terrain surface for the ground points on both sides
    ground = _ground(_curb_height, seed=9)
    ground = ground[np.abs(ground[:, 0] - 10.0) > 0.4]
    assert tin_deviation(tin, ground)["labels"]["sidewalk"]["rms_m"] < 0.01


def test_gentle_slope_creates_no_step_lines() -> None:
    slope = lambda xy: 0.02 * xy[:, 0] + 0.01 * xy[:, 1]  # noqa: E731
    sampler = _sampler(slope)
    assert st.detect_steps(REGIONS, sampler, config={}) == []
    tin = build_labeled_tin(_ground(slope), regions=REGIONS, dtm_sampler=sampler, steps=[], tolerance=0.02,
                            initial_spacing=2.0, max_vertices=20000)
    assert getattr(tin, "step_report", None) is None or tin.step_report["segments"] == 0
    assert len(tin.vertices) < 3000


def test_small_jump_below_threshold_stays_single_breakline() -> None:
    low = lambda xy: np.where(xy[:, 0] >= 10.0, 0.02, 0.0)  # noqa: E731
    low_curb = SimpleNamespace(points=np.array([[10.0, 0.0, 0.02], [10.0, 20.0, 0.02]]), foot_z=np.zeros(2), road_side=1)
    assert st.detect_steps(REGIONS, _sampler(low), curbs=[low_curb], config={}) == []   # a 2 cm kerb is no step


def test_manhole_in_road_lies_one_millimetre_below_cover() -> None:
    cover = 0.35
    flat = lambda xy: np.zeros(len(xy)) + 0.01 * xy[:, 0]  # noqa: E731
    holes = [(5.0, 10.0, cover, 0.6), (14.0, 4.0, 0.2, 0.7)]
    tin = build_labeled_tin(_ground(flat), regions=REGIONS, dtm_sampler=_sampler(flat), manholes=holes,
                            tolerance=0.02, initial_spacing=2.0, max_vertices=20000)
    from samcloud.outdoor_model.terrain import TinSampler
    sampler = TinSampler(tin.vertices, tin.triangles)
    for x, y, z, diameter in holes:
        angles = np.linspace(0, 2 * np.pi, 40, endpoint=False)
        radius = np.r_[0.0, np.full(40, 0.5 * diameter * 0.999)]
        pts = np.column_stack((x + radius * np.r_[0.0, np.cos(angles)], y + radius * np.r_[0.0, np.sin(angles)]))
        heights = sampler.sample(pts)
        assert np.all(np.isfinite(heights))
        assert np.all(heights <= z - 0.001 + 1e-9) and np.all(np.abs(heights - (z - 0.001)) < 0.002)
        ring = np.hypot(tin.vertices[:, 0] - x, tin.vertices[:, 1] - y)
        assert (np.abs(ring - 0.5 * diameter / np.cos(np.pi / 16)) < 1e-6).sum() == 16   # ring vertices are TIN vertices


def test_strips_have_own_label_stay_out_of_pavement_bodies_and_pass_rule_10() -> None:
    import shapely
    from shapely.ops import unary_union
    from samcloud.outdoor_model.terrain import build_terrain_elements

    sidewalk = box(6, 6, 14, 14)
    regions = {"sidewalk": [sidewalk]}
    sunken = lambda xy: np.where((xy[:, 0] > 6) & (xy[:, 0] < 14) & (xy[:, 1] > 6) & (xy[:, 1] < 14), -0.3, 0.0)  # noqa: E731
    sampler = _sampler(sunken)
    ring = np.array([[6.0, 6.0, 0.0], [14.0, 6.0, 0.0], [14.0, 14.0, 0.0], [6.0, 14.0, 0.0], [6.0, 6.0, 0.0]])
    curb = SimpleNamespace(points=ring[::-1], foot_z=np.full(5, -0.3), road_side=1)   # clockwise: the sunken side is on the left
    runs = st.detect_steps(regions, sampler, curbs=[curb], config={})
    assert runs and all(r.z_top.mean() > r.z_bottom.mean() for r in runs)
    tin = build_labeled_tin(_ground(sunken), regions=regions, dtm_sampler=sampler, steps=runs, tolerance=0.02,
                            initial_spacing=2.0, max_vertices=20000)
    labels = np.asarray(tin.labels)
    n_step = int((labels == TIN_LABELS["step"]).sum())
    assert n_step > 40
    assert len(tin.select("sidewalk").triangles) == int((labels == TIN_LABELS["sidewalk"]).sum())
    (terrain,), _ = build_terrain_elements(tin)
    assert terrain.evidence["step_triangles"] == n_step
    verts, faces = np.asarray(terrain.params.vertices), np.asarray(terrain.params.triangles)
    flagged = qa_cad._step_faces(verts, faces)
    assert flagged.sum() >= n_step - 4
    centres = verts[faces].mean(axis=1)
    inside = shapely.contains_xy(unary_union([sidewalk]), centres[:, 0], centres[:, 1])
    assert not np.any(inside & ~flagged)
    flat = np.array([[0, 0, 0.0], [2, 0, 0.0], [0, 2, 0.0]])
    assert not qa_cad._step_faces(flat, np.array([[0, 1, 2]])).any()


def _noisy_terrain_with_sunken_sidewalk():
    def height(xy):
        xy = np.asarray(xy, float).reshape(-1, 2)
        inside = (xy[:, 0] > 6) & (xy[:, 0] < 14) & (xy[:, 1] > 6) & (xy[:, 1] < 14)
        return np.where(inside, 0.20, 0.30) + 0.008 * np.sin(xy[:, 0] * 2.3 + xy[:, 1] * 1.1) + 0.004 * np.cos(xy[:, 1] * 5.0)
    return height


def test_pavement_without_curb_is_flush_with_noisy_higher_terrain_no_lip() -> None:
    height = _noisy_terrain_with_sunken_sidewalk()
    regions = {"sidewalk": [box(6, 6, 14, 14)]}
    assert st.detect_steps(regions, height, config={}) == []                      # no curb, no wall: no step at all
    plan = plan_flush(regions, height, [], config={})
    assert plan.lines and len(plan.edge_z) > 50
    tin = build_labeled_tin(_ground(height, seed=2), regions=regions, dtm_sampler=height, flush=plan,
                            tolerance=0.02, initial_spacing=2.0, max_vertices=20000)
    assert tin.flush_report["lines"] >= 1 and tin.flush_report["moved_vertices"] > 0
    assert (np.asarray(tin.labels) == TIN_LABELS["step"]).sum() == 0
    at = tin_sampler(tin)
    on = np.column_stack((np.linspace(6.5, 13.5, 60), np.full(60, 6.0)))
    inner, outer = at(on + [0.0, 0.005]), at(on - [0.0, 0.005])
    assert np.all(np.isfinite(inner)) and np.all(np.isfinite(outer))
    assert np.abs(outer - inner).max() < 0.005                                    # terrain edge == pavement edge (+-5 mm)
    for x in (8.0, 10.0, 12.0):                                                   # no lip: rises monotonically, never above its own height
        profile = at(np.column_stack((np.full(40, x), 6.0 - np.linspace(0.0, 2.0, 40))))
        assert np.all(np.diff(profile) > -0.004) and profile.max() < 0.315 and profile[0] < 0.23
    assert abs(float(np.mean(inner)) - 0.20) < 0.02                               # the edge height is the pavement side


def test_curb_edge_keeps_its_step_while_curbless_edges_stay_flush() -> None:
    height = _noisy_terrain_with_sunken_sidewalk()
    regions = {"sidewalk": [box(6, 6, 14, 14)]}
    curb = SimpleNamespace(points=np.array([[14.0, 14.0, 0.34], [14.0, 6.0, 0.34]]), foot_z=np.full(2, 0.20), road_side=-1)
    runs = st.detect_steps(regions, height, curbs=[curb], config={})
    assert len(runs) == 1 and runs[0].source == "curb" and abs(runs[0].mean_jump - 0.14) < 0.03
    plan = plan_flush(regions, height, runs, config={})
    assert all(not np.any(np.abs(line[:, 0] - 14.0) < 0.04) for line in plan.lines if abs(line[0, 1] - line[-1, 1]) > 0.5 and line[0, 0] > 13.9)
    tin = build_labeled_tin(_ground(height, seed=2), regions=regions, dtm_sampler=height, steps=runs, flush=plan,
                            tolerance=0.02, initial_spacing=2.0, max_vertices=20000)
    assert tin.step_report["segments"] > 20
    at = tin_sampler(tin)
    assert np.all(at(np.column_stack((np.full(20, 14.0), np.linspace(7, 13, 20)))) > 0.30)   # upper side (curb top) at the step
    south = np.column_stack((np.linspace(7, 13, 20), np.full(20, 6.0)))
    assert np.abs(at(south - [0, 0.005]) - at(south + [0, 0.005])).max() < 0.006             # the curbless side stays flush


def test_tin_sampler_returns_upper_side_at_a_step() -> None:
    sampler = _sampler(_curb_height)
    runs = st.detect_steps(REGIONS, sampler, curbs=[CURB], config={})
    tin = build_labeled_tin(_ground(_curb_height), regions=REGIONS, dtm_sampler=sampler, steps=runs, tolerance=0.02,
                            initial_spacing=2.0, max_vertices=20000)
    z = tin_sampler(tin)(np.array([[10.0, 5.0], [9.5, 5.0], [10.5, 5.0], [40.0, 40.0]]))
    assert abs(z[0] - 0.12) < 0.01 and abs(z[1]) < 0.01 and abs(z[2] - 0.12) < 0.01 and np.isnan(z[3])


def _ifc_rule_18(tin, tmp_path):
    from samcloud.bim.core.ifc import write_ifc
    from samcloud.drone import contracts as c
    from samcloud.drone.qa_cad import check_ifc
    from samcloud.outdoor_model import pavement as pv
    from samcloud.outdoor_model.scene import assemble_scene
    from samcloud.outdoor_model.terrain import build_terrain_elements

    polygons = [c.PavementPolygon(pavement_id=0, kind="sidewalk", polygon=box(6, 6, 14, 14))]
    cfg = c.load_drone_config()
    pave, _ = pv.build_pavement_elements(polygons, [], [], tin, config=cfg, topology_ready=True)
    terrain, _ = build_terrain_elements(tin, config=cfg)
    frame = c.LocalFrame(origin_lv95=(2_600_000.0, 1_200_000.0, 430.0))
    path = tmp_path / "flush.ifc"
    write_ifc(assemble_scene(pave, terrain, frame=frame), path)
    return {r["rule"]: r for r in check_ifc(path, cfg)}


def test_qa_rule_18_flags_a_lip_and_passes_when_flush(tmp_path) -> None:
    height = _noisy_terrain_with_sunken_sidewalk()
    regions = {"sidewalk": [box(6, 6, 14, 14)]}
    ground = _ground(height, seed=2)
    lip = build_labeled_tin(ground, regions=regions, dtm_sampler=height, tolerance=0.02, initial_spacing=2.0, max_vertices=20000)
    flush = build_labeled_tin(ground, regions=regions, dtm_sampler=height, flush=plan_flush(regions, height, [], config={}),
                              tolerance=0.02, initial_spacing=2.0, max_vertices=20000)
    # without the flush plan the terrain vertices on the outline take the (higher) terrain DTM height: a lip
    assert not _ifc_rule_18(lip, tmp_path)["18_terrain_flush_at_pavement"]["ok"]
    result = _ifc_rule_18(flush, tmp_path)["18_terrain_flush_at_pavement"]
    assert result["ok"], result
    assert result["value"]["samples"] > 20 and result["value"]["max_lip_m"] < 0.01
