"""Curb dedupe, cross-section (top, foot, gutter stone) and flush curbs, plus the manhole body top."""
from __future__ import annotations

import ifcopenshell
import ifcopenshell.geom
import numpy as np

from samcloud.bim.core.ifc import write_ifc
from samcloud.bim.core.model import ManholeElement, ManholeParams, Scene
from samcloud.drone import contracts as c
from samcloud.environment.curbs import curb_axes
from samcloud.outdoor_model import civil

CFG = c.load_drone_config()
FRAME = c.LocalFrame(origin_lv95=(2_600_000.0, 1_200_000.0, 430.0))


def _noise(rng, n, sxy=0.006, sz=0.004):
    return rng.normal(0, sxy, (n, 2)), rng.normal(0, sz, n)


def _strip(rng, x0, x1, y0, y1, z, density, code):
    n = int((x1 - x0) * (y1 - y0) * density)
    xy = np.column_stack([rng.uniform(x0, x1, n), rng.uniform(y0, y1, n)])
    dxy, dz = _noise(rng, n)
    return np.column_stack([xy + dxy, z + dz]), np.full(n, code, dtype=np.uint16)


def _scene_points(rng, *, top=0.12, with_gutter=True, length=20.0, curb_y=0.0, ground_density=300):
    """Curb along +x at y = curb_y. Road on the -y side (z = 0, optional curb-class gutter band), sidewalk on +y at top."""
    parts = [
        _strip(rng, 0, length, curb_y - 0.06, curb_y + 0.06, top, 5000, c.CURB),                   # curb top
        _strip(rng, 0, length, curb_y - 0.063, curb_y - 0.057, 0.0, 0, c.CURB),
    ]
    n = int(length * 300)
    face_z = rng.uniform(0.0, top, n)
    dxy, dz = _noise(rng, n)
    parts.append((np.column_stack([rng.uniform(0, length, n) + dxy[:, 0], curb_y - 0.06 + dxy[:, 1], face_z + dz]),
                  np.full(n, c.CURB, dtype=np.uint16)))
    if with_gutter:
        parts.append(_strip(rng, 0, length, curb_y - 0.36, curb_y - 0.06, 0.0, 600, c.CURB))      # gutter band, flat at foot level
        parts.append(_strip(rng, 0, length, curb_y - 2.0, curb_y - 0.36, 0.0, ground_density, c.ROAD))
    else:
        parts.append(_strip(rng, 0, length, curb_y - 2.0, curb_y - 0.06, 0.0, ground_density, c.ROAD))
    parts.append(_strip(rng, 0, length, curb_y + 0.06, curb_y + 2.0, top, ground_density, c.SIDEWALK))
    xyz = np.vstack([p for p, _ in parts if len(p)])
    classes = np.concatenate([k for p, k in parts if len(p)])
    return c.PointSet(xyz, classes)


def _export(elements, path):
    scene = Scene()
    for el in elements:
        scene.add(el)
    write_ifc(scene, path)
    f = ifcopenshell.open(str(path))
    settings = ifcopenshell.geom.settings()
    settings.set("use-world-coords", True)
    out = {}
    for product in f.by_type("IfcCivilElement"):
        shape = ifcopenshell.geom.create_shape(settings, product)
        out[product.ObjectType] = np.asarray(shape.geometry.verts).reshape(-1, 3)
    return out


def _road_side(axis) -> int:
    """The road lies on -y: right of a +x axis (-1), left of a -x axis (+1)."""
    return -1 if axis.points[-1, 0] > axis.points[0, 0] else 1


def test_duplicate_parallel_curb_axis_is_removed():
    rng = np.random.default_rng(31)
    real = _scene_points(rng, with_gutter=False)
    dup = _strip(rng, 0, 20, 0.30 - 0.06, 0.30 + 0.06, 0.12, 5000, c.CURB)
    ps = c.PointSet(np.vstack([real.xyz, dup[0]]), np.concatenate([real.classes, dup[1]]))
    report: dict = {}
    axes = curb_axes(ps, report=report)
    assert len(axes) == 1
    assert abs(axes[0].points[:, 2].mean() - 0.12) < 0.01


def test_separate_parallel_duplicate_counts_in_report():
    rng = np.random.default_rng(32)
    real = _scene_points(rng, with_gutter=False)
    dup = _strip(rng, 0, 20, 0.39, 0.51, 0.12, 5000, c.CURB)  # 0.45 m behind: separate cluster, still one curb
    ps = c.PointSet(np.vstack([real.xyz, dup[0]]), np.concatenate([real.classes, dup[1]]))
    report: dict = {}
    axes = curb_axes(ps, report=report)
    assert len(axes) == 1 and report["duplicates_removed"] == 1
    assert report["duplicate_length_m"] > 15.0


def test_curb_with_reveal_and_gutter_band_is_sectioned_and_exported(tmp_path):
    rng = np.random.default_rng(33)
    axes = curb_axes(_scene_points(rng), config=CFG)
    assert len(axes) == 1
    axis = axes[0]
    assert abs(axis.points[:, 2].mean() - 0.12) < 0.01                      # top z
    assert np.abs(axis.foot_z).max() < 0.01 and not axis.flush              # foot = road surface
    assert axis.road_side == _road_side(axis) and axis.gutter_measured
    assert abs(axis.gutter_width - 0.30) < 0.05
    elements, _ = civil.build_civil_elements(axes, config=CFG)
    curb = elements[0]
    meshes = _export(elements, tmp_path / "curb.ifc")
    body, gutter = meshes["Randstein"], meshes["Wasserstein"]
    assert abs(body[:, 2].max() - 0.12) < 0.01                               # the body ends at the top line
    assert abs(body[:, 2].min() - (0.0 - curb.params.embed)) < 0.02          # and hangs below it (not 0.25 m too high)
    assert abs(gutter[:, 2].max()) < 0.01                                    # gutter top at foot z
    assert abs(np.ptp(gutter[:, 1]) - (0.30)) < 0.05 + 0.0                   # gutter width
    assert gutter[:, 1].max() < 0.0                                          # on the road side
    entities = civil.to_dxf_entities(elements, FRAME, config=CFG)
    assert entities["S-GUTR"] and entities["S-CURB"]


def test_assumed_gutter_width_without_band():
    rng = np.random.default_rng(34)
    axes = curb_axes(_scene_points(rng, with_gutter=False), config=CFG)
    assert len(axes) == 1 and axes[0].road_side == _road_side(axes[0])
    assert not axes[0].gutter_measured and abs(axes[0].gutter_width - CFG["curb"]["gutter_width_m"]) < 1e-9


def test_flush_curb_does_not_stick_out(tmp_path):
    rng = np.random.default_rng(35)
    axes = curb_axes(_scene_points(rng, top=0.01, with_gutter=False), config=CFG)
    assert len(axes) == 1 and axes[0].flush
    assert axes[0].points[:, 2].max() <= 0.005
    elements, _ = civil.build_civil_elements(axes, config=CFG)
    assert elements[0].evidence["flush"] is True
    meshes = _export(elements, tmp_path / "flush.ifc")
    assert "Wasserstein" not in meshes
    assert meshes["Randstein"][:, 2].max() <= 0.005
    assert civil.to_dxf_entities(elements, FRAME, config=CFG)["S-CURB"]  # still written (single line)


def test_manhole_ifc_body_top_equals_cover_height(tmp_path):
    cover = 431.237
    manhole = ManholeElement(params=ManholeParams(center=(3.0, 4.0, cover), diameter=0.62, type="Schacht"))
    meshes_scene = Scene()
    meshes_scene.add(manhole)
    path = tmp_path / "mh.ifc"
    write_ifc(meshes_scene, path)
    f = ifcopenshell.open(str(path))
    settings = ifcopenshell.geom.settings()
    settings.set("use-world-coords", True)
    (chamber,) = f.by_type("IfcDistributionChamberElement")
    verts = np.asarray(ifcopenshell.geom.create_shape(settings, chamber).geometry.verts).reshape(-1, 3)
    assert abs(verts[:, 2].max() - cover) < 0.001


def test_clipped_curbs_get_exactly_one_s_curb_entity_each():
    from collections import defaultdict
    from types import SimpleNamespace
    from shapely.geometry import box
    from samcloud.drone import run_environment as run_env
    from samcloud.outdoor_model import pavement
    from samcloud.terrain.edges import BulgeLine, FittedCurb

    def axis(cid, y):
        pts = np.array([[x, y, 0.12 + 0.002 * x] for x in np.linspace(0.0, 30.0, 13)])
        return c.CurbAxis(cid, pts, c.Evidence(points=100), foot_z=pts[:, 2] - 0.12, road_side=1, gutter_width=0.3)

    axes = [axis(1, 0.0), axis(2, 5.0)]
    fit = SimpleNamespace(curbs=[
        FittedCurb(a.curb_id, BulgeLine(a.points[:, :2].copy(), np.zeros(len(a.points) - 1), {}), a.points[:, 2].copy(), a.points.copy(), False, "ok")
        for a in axes])
    elements, _ = civil.build_civil_elements(
        [c.CurbAxis(f.curb_id, f.dense_xyz, c.Evidence(points=100), foot_z=np.interp(f.dense_xyz[:, 0], axes[0].points[:, 0], axes[0].foot_z), road_side=1, gutter_width=0.3)
         for f in fit.curbs], config=CFG)
    # the boundary cuts curb 2 in the middle: two parts (curb-2, curb-2-p1) while the fitted export still has one line
    boundary = box(-1, -1, 31, 3.0).union(box(-1, 3.5, 12, 9)).union(box(18, 3.5, 31, 9))
    counts = defaultdict(int)
    clipped = run_env._clip_elements(elements, boundary, counts)
    curbs = [e for e in clipped if hasattr(e.params, "foot_z")]
    assert len(curbs) == 3 and {e.id for e in curbs} == {"curb-1", "curb-2", "curb-2-p1"}
    assert all(e.params.foot_z is not None and len(e.params.foot_z) == len(e.params.axis) for e in curbs)  # per-vertex data follows the clip
    skip, extra = civil.curbs_off_the_fit(clipped, fit.curbs, CFG)
    assert skip == {2} and {e.id for e in extra} == {"curb-2", "curb-2-p1"}
    fitted = pavement.curb_dxf_entities(fit.curbs, FRAME, config=CFG, exclude_ids=skip).get("S-CURB", [])
    from_elements = civil.to_dxf_entities(extra, FRAME, config=CFG).get("S-CURB", [])
    assert len(fitted) + len(from_elements) == len(curbs)
