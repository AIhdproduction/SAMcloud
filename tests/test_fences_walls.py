"""Fences (upright ribbon between terrain and top edge) and site walls (footprint follows the axis)."""
from __future__ import annotations

import ifcopenshell
import ifcopenshell.geom
import numpy as np
import shapely
from shapely.geometry import LineString, Polygon

from samcloud.bim.core.ifc import write_ifc
from samcloud.bim.core.model import Scene
from samcloud.drone import contracts as c
from samcloud.outdoor_model.site_walls import build_site_elements, fence_features, site_wall_features, to_dxf_entities

SLOPE = 0.08
FRAME = c.LocalFrame(origin_lv95=(2_600_000.0, 1_200_000.0, 430.0))


def _terrain(xy: np.ndarray) -> np.ndarray:
    return SLOPE * np.asarray(xy)[:, 0] + 0.5


def _fence_points(rng, polyline, height, per_m=250):
    """Vertical sheet with a top rail (30 % of the points) standing on the sloped terrain."""
    out = []
    for a, b in zip(polyline[:-1], polyline[1:]):
        a, b = np.asarray(a, float), np.asarray(b, float)
        n = int(np.hypot(*(b - a)) * per_m)
        t = rng.uniform(0, 1, n)
        xy = a + (b - a) * t[:, None] + rng.normal(0, 0.01, (n, 2))
        u = np.where(rng.random(n) < 0.3, 1.0 + rng.normal(0, 0.008, n), rng.uniform(0, 1, n))
        out.append(np.column_stack([xy, _terrain(xy) + height * u + rng.normal(0, 0.004, n)]))
    return np.vstack(out)


def _pointset(parts):
    xyz = np.vstack([p for p, _ in parts])
    return c.PointSet(xyz, np.concatenate([np.full(len(p), code, dtype=np.uint16) for p, code in parts]))


def _mesh_world(path, ifc_class):
    f = ifcopenshell.open(str(path))
    settings = ifcopenshell.geom.settings()
    settings.set("use-world-coords", True)
    out = []
    for product in f.by_type(ifc_class):
        shape = ifcopenshell.geom.create_shape(settings, product)
        out.append((product, np.asarray(shape.geometry.verts).reshape(-1, 3), np.asarray(shape.geometry.faces).reshape(-1, 3)))
    return out


def test_parallel_and_l_shaped_fences_follow_terrain_and_stay_upright(tmp_path):
    rng = np.random.default_rng(11)
    a = _fence_points(rng, [(0, 0), (10, 0)], 1.2)
    b = _fence_points(rng, [(0, 0.6), (10, 0.6)], 1.0)
    l_shape = _fence_points(rng, [(0, 6), (6, 6), (6, 11)], 1.5)
    fences = fence_features(_pointset([(a, c.FENCE), (b, c.FENCE), (l_shape, c.FENCE)]), _terrain)
    assert len(fences) == 3 and all(f.status == "ok" for f in fences)
    heights = sorted(round(f.height, 1) for f in fences)
    assert heights == [1.0, 1.2, 1.5]
    for fence in fences:
        xy = fence.axis[:, :2]
        assert np.abs(fence.bottom_z - _terrain(xy)).max() < 0.02
        true_h = {1.0: 1.0, 1.2: 1.2, 1.5: 1.5}[round(fence.height, 1)]
        assert np.abs(fence.axis[:, 2] - (_terrain(xy) + true_h)).max() < 0.03
    corner = next(f for f in fences if round(f.height, 1) == 1.5)
    assert np.hypot(corner.axis[:, 0] - 6, corner.axis[:, 1] - 6).min() < 0.15  # the L corner is a vertex
    parallel = [f for f in fences if round(f.height, 1) in (1.0, 1.2)]
    assert abs(np.median([f.axis[:, 1].mean() for f in parallel if round(f.height, 1) == 1.2]) - 0.0) < 0.05

    elements, qa = build_site_elements([], fences)
    assert not qa
    scene = Scene()
    for el in elements:
        scene.add(el)
    path = tmp_path / "fences.ifc"
    write_ifc(scene, path)
    products = _mesh_world(path, "IfcRailing")
    assert len(products) == 3
    for product, verts, faces in products:
        zrel = verts[:, 2] - _terrain(verts[:, :2])
        height = next(e.params.height for e in elements if e.id == product.Tag)
        # every vertex lies on the terrain or at the fence height above it: nothing is tilted or floating
        assert np.minimum(np.abs(zrel), np.abs(zrel - height)).max() < 0.05
        tri = verts[faces]
        volume = float(np.einsum("ij,ij->i", tri[:, 0], np.cross(tri[:, 1], tri[:, 2])).sum() / 6.0)
        assert volume > 0.0  # outward facing, closed


def test_fence_dxf_top_edge_and_optional_bottom_line():
    rng = np.random.default_rng(12)
    ps = _pointset([(_fence_points(rng, [(0, 0), (8, 0)], 1.1), c.FENCE)])
    fences = fence_features(ps, _terrain)
    elements, _ = build_site_elements([], fences)
    entities = to_dxf_entities(elements, FRAME)
    assert len(entities["S-FENC"]) == 1
    top = np.asarray(entities["S-FENC"][0]["points"])
    xy_local = top[:, :2] - np.array(FRAME.origin_lv95[:2])
    assert np.abs(top[:, 2] - FRAME.origin_lv95[2] - (_terrain(xy_local) + 1.1)).max() < 0.03
    cfg = c.load_drone_config()
    cfg["dxf"]["fence_bottom_line"] = True
    elements, _ = build_site_elements([], fences, config=cfg)
    assert len(to_dxf_entities(elements, FRAME)["S-FENC"]) == 2


def _wall_points(rng, polyline, thickness, height, per_m=250):
    out = []
    line = LineString(polyline)
    n = int(line.length * per_m)
    s = rng.uniform(0, line.length, n)
    base = shapely.get_coordinates(shapely.line_interpolate_point(line, s))
    ahead = shapely.get_coordinates(shapely.line_interpolate_point(line, np.minimum(s + 0.05, line.length)))
    back = shapely.get_coordinates(shapely.line_interpolate_point(line, np.maximum(s - 0.05, 0)))
    d = ahead - back
    d /= np.maximum(np.hypot(d[:, 0], d[:, 1]), 1e-9)[:, None]
    nrm = np.column_stack([-d[:, 1], d[:, 0]])
    xy = base + nrm * rng.uniform(-thickness / 2, thickness / 2, n)[:, None]
    z = np.where(rng.random(n) < 0.3, height + rng.normal(0, 0.005, n), rng.uniform(0, height, n))
    out.append(np.column_stack([xy, z]))
    return np.vstack(out)


def _footprint_of(path):
    (_p, verts, faces), = _mesh_world(path, "IfcWall")
    low = verts[:, 2] < verts[:, 2].min() + 1e-6
    tris = [Polygon(verts[f][:, :2]) for f in faces if low[f].all()]
    return shapely.union_all([t for t in tris if t.area > 1e-9]), verts


def test_bent_and_curved_walls_follow_their_axis_and_dxf_equals_ifc(tmp_path):
    rng = np.random.default_rng(13)
    thickness = 0.30
    l_axis = [(0, 0), (8, 0), (8, 6)]
    t = np.linspace(0, np.pi / 2, 30)
    arc_axis = [(20 + 6 * np.sin(a), 6 - 6 * np.cos(a)) for a in t]
    for axis in (l_axis, arc_axis):
        ps = _pointset([(_wall_points(rng, axis, thickness, 1.0), c.SITE_WALL)])
        walls = site_wall_features(ps, lambda xy: np.zeros(len(xy)))
        assert len(walls) == 1
        elements, qa = build_site_elements(walls, [])
        assert not qa
        scene = Scene()
        scene.add(elements[0])
        path = tmp_path / "wall.ifc"
        write_ifc(scene, path)
        footprint, _ = _footprint_of(path)
        truth = LineString(axis).buffer(thickness / 2, cap_style=2, join_style=2)
        iou = footprint.intersection(truth).area / footprint.union(truth).area
        assert iou >= 0.95, iou
        ring = np.asarray(to_dxf_entities(elements, FRAME)["S-WALL"][0]["points"])[:, :2] - np.array(FRAME.origin_lv95[:2])
        dxf_poly = Polygon(ring)
        assert dxf_poly.symmetric_difference(footprint).area / footprint.area < 1e-3


def test_wall_with_sloping_top_becomes_triangulated_solid(tmp_path):
    rng = np.random.default_rng(14)
    line = [(0, 0), (12, 0)]
    pts = _wall_points(rng, line, 0.25, 1.0)
    pts[:, 2] *= 1.0 + 0.6 * pts[:, 0] / 12.0  # top rises from 1.0 to 1.6 m
    walls = site_wall_features(_pointset([(pts, c.SITE_WALL)]), lambda xy: np.zeros(len(xy)))
    elements, _ = build_site_elements(walls, [])
    assert elements[0].params.top_z_axis is not None
    scene = Scene()
    scene.add(elements[0])
    path = tmp_path / "sloped.ifc"
    write_ifc(scene, path)
    _fp, verts = _footprint_of(path)
    near_start, near_end = verts[verts[:, 0] < 1.0, 2].max(), verts[verts[:, 0] > 11.0, 2].max()
    assert near_end - near_start > 0.35


def _fence_with_gap(rng, gap):
    pts = _fence_points(rng, [(0, 0), (10, 0)], 1.2)
    keep = (pts[:, 0] < 4.0) | (pts[:, 0] > 4.0 + gap)
    return pts[keep]


def test_collinear_fragments_merge_but_real_openings_stay_gaps():
    rng = np.random.default_rng(15)
    report: dict = {}
    merged = fence_features(_pointset([(_fence_with_gap(rng, 0.7), c.FENCE)]), _terrain, report=report)
    assert len(merged) == 1 and report == {"traced": 2, "merged": 1}
    assert merged[0].axis[:, 0].max() - merged[0].axis[:, 0].min() > 9.5
    report = {}
    gate = fence_features(_pointset([(_fence_with_gap(rng, 1.8), c.FENCE)]), _terrain, report=report)
    assert len(gate) == 2 and report == {"traced": 2, "merged": 2}
    # different heights or directions are never merged
    low = _fence_points(rng, [(0, 0), (4, 0)], 1.2)
    high = _fence_points(rng, [(4.5, 0), (10, 0)], 2.0)
    assert len(fence_features(_pointset([(low, c.FENCE), (high, c.FENCE)]), _terrain)) == 2


def test_fence_on_plinth_keeps_its_own_lower_edge():
    rng = np.random.default_rng(16)
    pts = _fence_points(rng, [(0, 0), (10, 0)], 1.2)
    pts[:, 2] += 0.6  # the fence stands on a plinth: its points never reach the terrain
    (fence,) = fence_features(_pointset([(pts, c.FENCE)]), _terrain)
    terrain = _terrain(fence.axis[:, :2])
    assert (fence.bottom_z - terrain).min() > 0.45          # not pulled down onto the terrain
    assert abs(fence.height - 1.2) < 0.1 and (fence.axis[:, 2] - fence.bottom_z).min() >= 0.2


def test_site_wall_base_follows_the_terrain(tmp_path):
    rng = np.random.default_rng(17)
    pts = _wall_points(rng, [(0, 0), (12, 0)], 0.25, 1.0)
    pts[:, 2] += _terrain(pts[:, :2])  # wall stands on the slope, constant height
    (wall,) = site_wall_features(_pointset([(pts, c.SITE_WALL)]), _terrain)
    assert wall.base_z_axis is not None and abs(wall.base_z_axis[-1] - wall.base_z_axis[0]) > 0.7
    elements, _ = build_site_elements([wall], [])
    assert elements[0].params.base_z_axis is not None
    scene = Scene()
    scene.add(elements[0])
    path = tmp_path / "base.ifc"
    write_ifc(scene, path)
    _fp, verts = _footprint_of(path)
    low = verts[verts[:, 2] < verts[:, 2].min() + 1.0]
    z_start = verts[verts[:, 0] < 0.5, 2].min()
    z_end = verts[verts[:, 0] > 11.5, 2].min()
    assert z_end - z_start > 0.7
