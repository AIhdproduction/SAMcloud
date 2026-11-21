"""Tests for samcloud.environment.image_edges (synthetic noisy ortho, jittered class-raster edges)."""
import math

import numpy as np
import shapely
from shapely.geometry import LineString, Polygon, box

from samcloud.environment import topology as tp
from samcloud.environment.image_edges import drop_short_edges, refine_partition
from samcloud.environment.ortho_rgb import OrthoRGB
from samcloud.terrain.edges import BulgeLine, arc_geometry, fit_surfaces

CELL = 0.03
ROAD, SIDEWALK, TERRAIN = (95, 95, 100), (150, 148, 140), (70, 120, 60)


def _truth_chain() -> np.ndarray:
    """Straight run (0,8)-(14,8), arc of radius 6 around (14,14) over 45 degrees, tangent straight run to x = 20."""
    straight = np.column_stack((np.linspace(0, 14, 141), np.full(141, 8.0)))
    ang = np.linspace(-math.pi / 2, -math.pi / 4, 48)[1:]
    arc = np.column_stack((14 + 6 * np.cos(ang), 14 + 6 * np.sin(ang)))
    end = arc[-1]
    steps = np.linspace(0.0, (20.0 - end[0]) * math.sqrt(2.0), 20)[1:]
    tail = np.column_stack((end[0] + steps / math.sqrt(2.0), end[1] + steps / math.sqrt(2.0)))
    return np.vstack((straight, arc, tail))


def _paint(shape_polys: dict, colours: dict, seed: int, noise: float = 3.0, size=(-2.0, 22.0)) -> OrthoRGB:
    lo, hi = size
    n = int(round((hi - lo) / CELL))
    xs = lo + (np.arange(n) + 0.5) * CELL
    ys = hi - (np.arange(n) + 0.5) * CELL
    gx, gy = np.meshgrid(xs, ys)
    img = np.empty((n, n, 3), float)
    img[:] = colours["terrain"]
    for name, poly in shape_polys.items():
        inside = shapely.contains_xy(poly, gx, gy)
        img[inside] = colours[name]
    rng = np.random.default_rng(seed)
    img += rng.normal(0.0, noise, img.shape)
    return OrthoRGB(np.clip(img, 0, 255).astype(np.uint8), np.ones((n, n), bool), lo, hi, CELL)


def _scene(seed=3, jitter=0.2):
    truth = _truth_chain()
    d = np.r_[0.0, np.cumsum(np.hypot(*np.diff(truth, axis=0).T))]
    tang = np.gradient(truth, axis=0)
    tang /= np.hypot(*tang.T)[:, None]
    nrm = np.column_stack((-tang[:, 1], tang[:, 0]))
    rng = np.random.default_rng(seed)
    taper = np.clip(np.minimum(d, d[-1] - d) / 1.5, 0, 1)
    wave = jitter * np.sin(2 * np.pi * d / 9.0 + 0.7) * taper
    noisy = truth + (wave + rng.normal(0, 0.01, len(d)) * taper)[:, None] * nrm
    noisy = noisy[::5].copy()
    noisy[-1] = truth[-1]
    sidewalk = Polygon(np.vstack((noisy, [(20, 20), (0, 20)])))
    road = box(0, 0, 20, 20).difference(sidewalk)
    sw_truth = Polygon(np.vstack((truth, [(20, 20), (0, 20)])))
    rd_truth = box(0, 0, 20, 20).difference(sw_truth)
    ortho = _paint({"road": rd_truth, "sidewalk": sw_truth}, {"road": ROAD, "sidewalk": SIDEWALK, "terrain": TERRAIN}, seed)
    return truth, road, sidewalk, ortho


def _shared(road_faces, sidewalk_faces):
    a = shapely.union_all(road_faces).boundary
    b = shapely.union_all(sidewalk_faces).boundary
    return a.intersection(b)


def _arc_radius(chain: BulgeLine) -> float:
    k = int(np.flatnonzero(np.abs(chain.bulges) > 1e-9)[0])
    return float(arc_geometry(chain.pts[k], chain.pts[k + 1], chain.bulges[k])[1])


def test_refined_edge_within_3cm_straight_line_and_arc():
    truth, road, sidewalk, ortho = _scene()
    jitter_err = max(shapely.distance(shapely.points(np.asarray(sidewalk.exterior.coords)[:-3]), LineString(truth)))
    assert jitter_err > 0.12  # the class-raster edge really is off
    res = refine_partition({"road": [road], "sidewalk": [sidewalk]}, ortho)
    assert res.report["refined"] >= 1 and res.report["mean_shift_m"] > 0.03
    road_f, sw_f = res.regions["road"][0], res.regions["sidewalk"][0]
    assert len(road_f) == 1 and len(sw_f) == 1
    shared = _shared(road_f, sw_f)
    parts = [g for g in getattr(shared, "geoms", [shared]) if g.length > 1.0]
    pts = np.vstack([np.asarray([g.interpolate(t).coords[0] for t in np.linspace(0.0, g.length, int(g.length / 0.05))])
                     for g in parts])
    keep = (pts[:, 0] > 0.5) & (pts[:, 0] < 19.5)
    dist = shapely.distance(shapely.points(pts[keep]), LineString(truth))
    assert dist.max() < 0.03
    # line / arc structure after the final fit: the straight run is one line, the curve one arc of r = 6 m
    fit = fit_surfaces({"road": list(road_f), "sidewalk": list(sw_f)})
    inner = [ch for ch in fit.chains if ch.arcs]
    assert len(inner) == 1 and inner[0].arcs == 1 and 2 <= inner[0].lines <= 3
    assert abs(_arc_radius(inner[0]) - 6.0) < 0.6
    assert abs(inner[0].pts[0][1] - 8.0) < 0.03 and abs(inner[0].pts[1][1] - 8.0) < 0.03


NO_CLEAN = {"edges": {"clean": {"enabled": False}}}


def test_low_contrast_or_missing_colour_keeps_edge():
    _truth, road, sidewalk, ortho = _scene()
    flat = _paint({}, {"terrain": (110, 110, 110)}, seed=5, noise=3.0)
    res = refine_partition({"road": [road], "sidewalk": [sidewalk]}, flat, config=NO_CLEAN)
    assert res.report.get("fallback")
    assert res.regions["road"][0][0].equals(road) and res.regions["sidewalk"][0][0].equals(sidewalk)
    none = refine_partition({"road": [road], "sidewalk": [sidewalk]}, None, config=NO_CLEAN)
    assert none.regions["road"][0][0].equals(road)
    cfg = {"edges": {"image": {"enabled": False}, "clean": {"enabled": False}}}
    off = refine_partition({"road": [road], "sidewalk": [sidewalk]}, ortho, config=cfg)
    assert off.regions["sidewalk"][0][0].equals(sidewalk)


def test_shared_edge_identical_and_no_gap_or_overlap():
    _truth, road, sidewalk, ortho = _scene(seed=11)
    res = refine_partition({"road": [road], "sidewalk": [sidewalk]}, ortho)
    road_f, sw_f = res.regions["road"][0], res.regions["sidewalk"][0]
    shared = _shared(road_f, sw_f)
    assert shared.length > 20.0
    verts_r = {tuple(np.round(p, 6)) for f in road_f for p in f.exterior.coords}
    verts_s = {tuple(np.round(p, 6)) for f in sw_f for p in f.exterior.coords}
    for g in getattr(shared, "geoms", [shared]):
        for p in g.coords:
            assert tuple(np.round(p, 6)) in verts_r and tuple(np.round(p, 6)) in verts_s
    rep = tp.gap_overlap_report(road_f, sw_f)
    assert rep["gap_cm2"] < 1.0 and rep["overlap_cm2"] < 1.0 and rep["ok"]
    assert abs(sum(f.area for f in road_f) + sum(f.area for f in sw_f) - 400.0) < 1.5


def test_t_junction_keeps_a_single_node():
    # road (south), sidewalk (north-west), path (north-east) meet at T = (10, 8); class-raster edges are off by 10-15 cm
    ortho = _paint({"road": box(0, 0, 20, 8), "sidewalk": box(0, 8, 10, 20), "path": box(10, 8, 20, 20)},
                   {"road": ROAD, "sidewalk": SIDEWALK, "path": (120, 90, 70), "terrain": TERRAIN}, seed=21)
    sw = Polygon([(0, 8.15), (10.1, 8.15), (10.1, 20), (0, 20)])
    pa = Polygon([(10.1, 8.1), (20, 8.1), (20, 20), (10.1, 20)])
    rd = box(0, 0, 20, 20).difference(sw).difference(pa)
    res = refine_partition({"road": [rd], "sidewalk": [sw], "path": [pa]}, ortho)
    faces = [f for name in ("road", "sidewalk", "path") for fs in res.regions[name] for f in fs]
    assert len(faces) == 3 and res.report["refined"] >= 2
    node = np.array([10.0, 8.0])
    found = []
    for f in faces:
        ring = np.asarray(f.exterior.coords)[:-1]
        near = ring[np.hypot(*(ring - node).T) < 0.08]
        assert len(near) == 1, near
        found.append(near[0])
    assert all(np.allclose(found[0], q, atol=1e-6) for q in found) and np.hypot(*(found[0] - node)) < 0.04
    names = {"road": res.regions["road"], "sidewalk": res.regions["sidewalk"], "path": res.regions["path"]}
    for a, b in (("road", "sidewalk"), ("road", "path"), ("sidewalk", "path")):
        rep = tp.gap_overlap_report([f for fs in names[a] for f in fs], [f for fs in names[b] for f in fs])
        assert rep["ok"], (a, b, rep)


def test_curb_chain_is_never_moved():
    _truth, road, sidewalk, ortho = _scene(seed=5, jitter=0.0)
    ring = np.asarray(sidewalk.exterior.coords)[:-3, :2]

    class Curb:  # minimal CurbAxis stand-in
        points = np.column_stack((ring, np.zeros(len(ring))))

    before = sidewalk.exterior.intersection(road.boundary)
    res = refine_partition({"road": [road], "sidewalk": [sidewalk]}, ortho, [Curb()])
    shared = _shared(res.regions["road"][0], res.regions["sidewalk"][0])
    assert shared.hausdorff_distance(before) < 0.02


def test_drop_short_edges_removes_zigzag_but_keeps_corners():
    pts = np.array([[0, 0], [5, 0.01], [5.2, 0.03], [10, 0.0], [10, 5.0]], float)
    line = drop_short_edges(BulgeLine(pts, np.zeros(4)), 0.5, 0.05)
    assert len(line.pts) == 3 and np.allclose(line.pts[1], [10, 0.0])
    assert not any(np.allclose(p, [5.2, 0.03]) for p in line.pts)


def test_wavy_class_boundary_becomes_one_straight_line():
    xs = np.linspace(0.0, 20.0, 201)
    wave = 0.15 * np.sin(2 * np.pi * xs / 6.0) * np.clip(np.minimum(xs, 20 - xs) / 1.0, 0, 1)
    ring = np.column_stack((xs, 8.0 + wave))
    sidewalk = Polygon(np.vstack((ring, [(20, 20), (0, 20)])))
    road = box(0, 0, 20, 20).difference(sidewalk)
    ortho = _paint({"road": box(0, 0, 20, 8), "sidewalk": box(0, 8, 20, 20)},
                   {"road": ROAD, "sidewalk": SIDEWALK, "terrain": TERRAIN}, seed=31)
    res = refine_partition({"road": [road], "sidewalk": [sidewalk]}, ortho)
    assert res.report["refined"] >= 1
    shared = _shared(res.regions["road"][0], res.regions["sidewalk"][0])
    parts = [g for g in getattr(shared, "geoms", [shared]) if g.length > 5.0]
    assert len(parts) == 1 and len(parts[0].coords) == 2          # one straight segment
    pts = np.asarray([parts[0].interpolate(t).coords[0] for t in np.linspace(0.0, parts[0].length, 100)])
    assert np.abs(pts[:, 1] - 8.0).max() < 0.03


def test_dominant_orientations_find_axes():
    from samcloud.environment.image_edges import dominant_orientations
    pairs = [(math.radians(0.3), 30.0), (math.radians(-0.2), 20.0), (math.radians(90.4), 25.0), (math.radians(37.0), 3.0)]
    dirs = [math.degrees(d) for d in dominant_orientations(pairs)]
    assert len(dirs) == 2 and min(abs(d - 0.0) % 180 for d in dirs) < 0.5 or min(abs(d - 180.0) for d in dirs) < 0.5
    assert min(abs(d - 90.2) for d in dirs) < 0.6


def _spiky_scene():
    xs = np.linspace(0.0, 20.0, 401)
    rng = np.random.default_rng(4)
    taper = np.clip(np.minimum(xs, 20 - xs) / 1.0, 0, 1)
    y = 8.0 + (0.10 * np.sin(2 * np.pi * xs / 5.0) + rng.normal(0, 0.01, len(xs))) * taper
    for x0, h in ((5.0, 0.5), (11.0, -0.45), (15.5, 0.6)):      # three narrow spikes
        k = int(np.argmin(np.abs(xs - x0)))
        y[k] += h
        y[k - 1] += h * 0.4
        y[k + 1] += h * 0.4
    sidewalk = Polygon(np.vstack((np.column_stack((xs, y)), [(20, 20), (0, 20)])))
    road = box(0, 0, 20, 20).difference(sidewalk)
    return road, sidewalk


def test_clean_without_image_removes_spikes_and_straightens():
    road, sidewalk = _spiky_scene()
    res = refine_partition({"road": [road], "sidewalk": [sidewalk]}, None)
    assert res.report["spikes_removed"] >= 3 and res.report["cleaned"] >= 1
    road_f, sw_f = res.regions["road"][0], res.regions["sidewalk"][0]
    shared = _shared(road_f, sw_f)
    parts = [g for g in getattr(shared, "geoms", [shared]) if g.length > 5.0]
    assert len(parts) == 1 and len(parts[0].coords) == 2
    pts = np.asarray([parts[0].interpolate(t).coords[0] for t in np.linspace(0.0, parts[0].length, 100)])
    assert np.abs(pts[:, 1] - 8.0).max() < 0.05
    assert max(abs(sum(f.area for f in sw_f) - sidewalk.area), 0) / sidewalk.area < 0.05
    rep = tp.gap_overlap_report(road_f, sw_f)
    assert rep["gap_cm2"] < 1.0 and rep["overlap_cm2"] < 1.0


def test_bulge_ring_keeps_arcs_after_close_slivers():
    from samcloud.environment.image_edges import close_slivers
    ang = np.linspace(-math.pi / 2, 0.0, 60)
    arc = np.column_stack((14 + 6 * np.cos(ang), 14 + 6 * np.sin(ang)))
    chain = np.vstack(([(0.0, 8.0)], arc, [(20.0, 20.0), (0.0, 20.0)]))
    sidewalk = Polygon(chain)
    road = box(0, 0, 20, 20).difference(sidewalk).buffer(-0.0)
    road = Polygon(np.vstack(([(0.0, 0.0), (20.0, 0.0), (20.0, 14.0)], arc[::-1], [(0.0, 8.0)])))
    fit = fit_surfaces({"road": [road], "sidewalk": [sidewalk]})
    assert "fallback" not in fit.report
    # a 3 cm sliver between the faces, as refitting leaves it
    sw_face = fit.regions["sidewalk"][0][0]
    gap = box(2.0, 7.97, 8.0, 8.0)
    fit.regions["sidewalk"][0][0] = sw_face.difference(box(2.0, 7.99, 8.0, 8.05))
    before = fit.bulge_ring(np.asarray(fit.regions["road"][0][0].exterior.coords)).arcs
    assert before >= 1
    close_slivers(fit.regions)
    for name in ("road", "sidewalk"):
        ring = np.asarray(fit.regions[name][0][0].exterior.coords)
        assert fit.bulge_ring(ring).arcs >= 1, name
    assert gap.area > 0


def test_clean_never_moves_curb_chains():
    road, sidewalk = _spiky_scene()
    ring = np.asarray(sidewalk.exterior.coords)[:-3, :2]

    class Curb:
        points = np.column_stack((ring, np.zeros(len(ring))))

    res = refine_partition({"road": [road], "sidewalk": [sidewalk]}, None, [Curb()])
    shared = _shared(res.regions["road"][0], res.regions["sidewalk"][0])
    assert shared.hausdorff_distance(sidewalk.exterior.intersection(road.boundary)) < 0.02
