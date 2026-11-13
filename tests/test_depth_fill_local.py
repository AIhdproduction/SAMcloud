"""Round 5: local, edge-aware alignment of the Depth-Anything prior, sparse anchors, plane completion."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from samcloud.colmap import image_edges as ie
from samcloud.colmap.depth_fill import DEFAULTS, PriorRefiner, _scaled, fill_from_depth_prior
from samcloud.colmap.plane_fill import View, write_colmap_array
from samcloud.colmap.surface_fusion import derive_units_per_metre

SIZE = 480
FOCAL = 300.0
WALL_CLASS = 2


def _view(index: int, center: np.ndarray, forward=(0.0, 1.0, 0.0)) -> View:
    f = np.asarray(forward, float)
    f = f / np.linalg.norm(f)
    right = np.cross(f, [0.0, 0.0, 1.0])
    right /= np.linalg.norm(right)
    down = np.cross(f, right)
    rotation = np.vstack([right, down, f])
    return View(index, f"cam_{index}.png", rotation, -rotation @ center, FOCAL, FOCAL, SIZE / 2, SIZE / 2, SIZE, SIZE)


def _views(unit: float, count: int = 8, forward=(0.0, 1.0, 0.0), height: float = 1.3) -> list[View]:
    out = []
    for i in range(count):  # off-grid positions, >= 0.3 m apart
        center = np.array([-1.0 + 0.37 * i + 0.011 * i * i, 0.013 * i - 0.2, height + 0.021 * i]) * unit
        out.append(_view(i, center, forward))
    return out


def _rays(view: View) -> np.ndarray:
    v, u = np.mgrid[0:SIZE, 0:SIZE].astype(float) + 0.5
    cam = np.stack(((u - view.cx) / view.fx, (v - view.cy) / view.fy, np.ones_like(u)), axis=-1)
    return cam @ view.rotation  # world directions, z-depth parameterised (camera z = 1)


def _hit(view: View, normal, offset: float):
    """Camera depth and world point of the ray/plane hit (plane n.X + o = 0)."""
    rays = _rays(view)
    n = np.asarray(normal, float)
    denom = rays @ n
    with np.errstate(divide="ignore", invalid="ignore"):
        t = -(view.center @ n + offset) / denom
    t = np.where(np.isfinite(t) & (t > 0), t, 0.0)
    return t, view.center + rays * t[..., None]


def _wall_with_niche(view: View, unit: float, y0: float = 3.0, niche=(-1.2, 1.2, 0.5), z_range=(-5.0, 5.0)):
    """Corridor wall at y0 with a niche (x range) whose back wall lies ``depth`` further; world arrays."""
    t0, p0 = _hit(view, (0, 1, 0), -y0 * unit)
    t1, p1 = _hit(view, (0, 1, 0), -(y0 + niche[2]) * unit)
    in_niche = (p0[..., 0] > niche[0] * unit) & (p0[..., 0] < niche[1] * unit)
    depth = np.where(in_niche, t1, t0)
    world = np.where(in_niche[..., None], p1, p0)
    return depth.astype(np.float32), world, in_niche


def _write_stereo(tmp_path: Path, views, depths) -> Path:
    stereo = tmp_path / "stereo"
    (stereo / "depth_maps").mkdir(parents=True, exist_ok=True)
    for view, depth in zip(views, depths):
        write_colmap_array(stereo / "depth_maps" / f"{view.name}.geometric.bin", depth.astype(np.float32))
    return stereo


def _anchors(view: View, depth: np.ndarray, mask: np.ndarray, count: int, rng) -> tuple[np.ndarray, np.ndarray]:
    rows, cols = np.nonzero(mask & (depth > 0))
    pick = rng.choice(len(rows), min(count, len(rows)), replace=False)
    uv = np.column_stack((cols[pick] + rng.uniform(0, 1, len(pick)), rows[pick] + rng.uniform(0, 1, len(pick))))
    return uv.astype(np.float32), depth[rows[pick], cols[pick]].astype(np.float32)


def _settings(**extra):
    return {**DEFAULTS, "local_alignment": True, "use_edges": True, "use_sparse": True, "halo_filter": True,
            "anchor_rule": False, "max_neighbor_distance_m": 5.0, "device": "numpy", **extra}


def _central(depth: np.ndarray) -> np.ndarray:
    return depth[SIZE // 2 - 40:SIZE // 2 + 40, SIZE // 2 - 40:SIZE // 2 + 40]


def test_local_alignment_pulls_niche_back_wall_forward(tmp_path: Path) -> None:
    unit = 1.0
    rng = np.random.default_rng(3)
    views = _views(unit)
    depths, priors, anchors = [], {}, {}
    for view in views:
        depth, world, in_niche = _wall_with_niche(view, unit)
        valid = depth.copy()
        valid[in_niche] = 0.0  # COLMAP measured the corridor wall only
        depths.append(valid)
        wrong = np.where(in_niche, depth * 1.09, depth)  # Depth-Anything places the niche back wall ~0.3 m too deep
        priors[view.name] = (wrong * (1 + rng.normal(0, 0.003, wrong.shape))).astype(np.float32)
        anchors[view.name] = _anchors(view, depth, in_niche, 400, rng)  # reliable sparse points on the niche back wall
    stereo = _write_stereo(tmp_path, views, depths)
    refiner = PriorRefiner(views, stereo, priors.get, _settings(use_edges=False), anchors=anchors)

    for view in views[:3]:
        truth, _w, in_niche = _wall_with_niche(view, unit)
        inner = np.zeros_like(in_niche)
        inner[60:420, :] = in_niche[60:420, :]
        inner &= np.roll(in_niche, 70, axis=1) & np.roll(in_niche, -70, axis=1)  # a tile away from the niche edges
        raw = priors[view.name]
        local = refiner.depth(view.name)
        assert inner.sum() > 5000
        assert np.abs(raw[inner] / truth[inner] - 1).mean() > 0.07  # the global prior is clearly off in the niche
        trusted = inner & (local > 0)
        assert trusted.sum() > 0.9 * inner.sum()
        assert np.abs(local[trusted] / truth[trusted] - 1).mean() < 0.025


def test_da_only_agreement_without_colmap_anchor_rejected(tmp_path: Path) -> None:
    unit = 1.0
    views = _views(unit)
    depths, priors = [], {}
    rng = np.random.default_rng(4)
    for view in views:
        depth, _world, _niche = _wall_with_niche(view, unit, niche=(0, 0, 0))
        depths.append(np.zeros_like(depth))  # COLMAP measured nothing: white wall everywhere
        priors[view.name] = (depth * 1.12 * (1 + rng.normal(0, 0.003, depth.shape))).astype(np.float32)  # same bias in all views
    stereo = _write_stereo(tmp_path, views, depths)
    base = {**DEFAULTS, "local_alignment": False, "device": "numpy", "max_neighbor_distance_m": 5.0}

    without_rule, *_ = fill_from_depth_prior(views, stereo, priors.get, np.empty((0, 3)), {**base, "anchor_rule": False})
    with_rule, _rgb, _cls, _votes, report = fill_from_depth_prior(
        views, stereo, priors.get, np.empty((0, 3)), {**base, "anchor_rule": True})

    assert len(without_rule) > 500  # the views agree with each other (same bias), as before
    assert len(with_rule) == 0 and report["rejected_anchor"] > 500


def _hole_scene(tmp_path: Path, unit: float, *, niche=None, plane_bias: float = 1.12, with_sparse: bool = True):
    """Textureless wall: COLMAP measured a ring around the hole only; the prior is biased inside the hole."""
    rng = np.random.default_rng(5)
    views = _views(unit)
    depths, priors, anchors, labels, truths = [], {}, {}, {}, {}
    for view in views:
        depth, world, in_niche = _wall_with_niche(view, unit, niche=niche or (0, 0, 0))
        x = world[..., 0] / unit
        hole = (np.abs(x + 0.2) < 1.3) & (np.abs(world[..., 2] / unit - 1.3) < 1.0)
        valid = np.where(hole, 0.0, depth)
        depths.append(valid.astype(np.float32))
        priors[view.name] = (np.where(hole, depth * plane_bias, depth) * (1 + rng.normal(0, 0.003, depth.shape))).astype(np.float32)
        labels[view.name] = np.full((SIZE, SIZE), WALL_CLASS, np.uint8)
        truths[view.name] = (depth, world, hole, in_niche)
        if with_sparse:
            ring = (np.abs(x + 0.2) < 1.6) & (np.abs(world[..., 2] / unit - 1.3) < 1.3)
            anchors[view.name] = _anchors(view, depth, ring, 160, rng)
    return views, _write_stereo(tmp_path, views, depths), priors, anchors, labels, truths


def test_textureless_wall_plane_completion(tmp_path: Path) -> None:
    unit = 1.0
    views, stereo, priors, anchors, labels, truths = _hole_scene(tmp_path, unit)
    settings = _settings(anchor_rule=True, plane_completion=True, plane_label_ids=[WALL_CLASS], use_edges=False)

    xyz, _rgb, _cls, _votes, report = fill_from_depth_prior(
        views, stereo, priors.get, np.empty((0, 3)), settings, label_reader=labels.get, anchors=anchors)

    hole = (np.abs(xyz[:, 0] + 0.2) < 1.1) & (np.abs(xyz[:, 2] - 1.3) < 0.8)
    assert np.count_nonzero(hole) > 400
    assert np.all(np.abs(xyz[hole][:, 1] - 3.0) < 0.04)  # the plane wins over the 12 % too deep prior
    assert report["alignment"]["plane_views"] > 0


def test_plane_fit_does_not_fill_niche(tmp_path: Path) -> None:
    unit = 1.0
    views, stereo, priors, anchors, labels, truths = _hole_scene(tmp_path, unit, niche=(-1.0, 0.6, 0.5))
    settings = _settings(anchor_rule=True, plane_completion=True, plane_label_ids=[WALL_CLASS], use_edges=False)

    xyz, *_rest = fill_from_depth_prior(
        views, stereo, priors.get, np.empty((0, 3)), settings, label_reader=labels.get, anchors=anchors)

    in_niche = (xyz[:, 0] > -0.8) & (xyz[:, 0] < 0.4) & (np.abs(xyz[:, 2] - 1.3) < 0.8)
    assert np.count_nonzero(in_niche) > 100
    assert not np.any(np.abs(xyz[in_niche][:, 1] - 3.0) < 0.15)  # no corridor plane plastered over the niche
    assert np.median(np.abs(xyz[in_niche][:, 1] - 3.5)) < 0.06  # the niche keeps its own, deeper plane


def test_sparse_anchor_filter_rejects_high_error_and_short_tracks() -> None:
    view = _view(0, np.zeros(3))

    def point(xyz, error, track):
        elements = [SimpleNamespace(image_id=1, point2D_idx=0)]  # this image observes the point once
        return SimpleNamespace(xyz=np.asarray(xyz, float), error=error, track=SimpleNamespace(length=lambda n=track: n, elements=elements))

    points = {
        1: point([0.1, 3.0, 0.2], 0.6, 5),   # good
        2: point([0.3, 3.5, -0.1], 1.2, 3),  # good (limits)
        3: point([0.2, 3.0, 0.0], 2.4, 6),   # reprojection error too high (switch / handle)
        4: point([-0.2, 3.0, 0.1], 0.5, 2),  # track too short
        5: point([0.0, 3.0, 0.0], 1.6, 2),   # both
    }
    reconstruction = SimpleNamespace(points3D=points, images={1: SimpleNamespace(name=view.name)})

    result = ie.load_sparse_anchors(None, [view], 1.5, 3, reconstruction=reconstruction)

    uv, z = result[view.name]
    assert len(z) == 2 and np.allclose(sorted(z), [3.0, 3.5])


def test_edge_aware_alignment_does_not_bleed_across_niche_edge() -> None:
    h = w = 240
    truth = np.full((h, w), 3.0, np.float32)
    truth[:, 100:140] = 3.5  # narrow niche, 0.5 deeper
    prior = truth.copy()
    prior[:, 100:140] = 3.5 * 1.09  # Depth-Anything puts the niche back wall too deep
    geometric = np.where((np.arange(w)[None, :] < 100) | (np.arange(w)[None, :] >= 140), truth, 0.0).astype(np.float32)
    rng = np.random.default_rng(6)
    rows = rng.integers(0, h, 40)
    cols = rng.integers(104, 136, 40)
    sparse = (rows, cols, truth[rows, cols])
    strong = np.zeros((h, w), bool)
    strong[:, 98:101] = True  # corner lines of the niche
    strong[:, 139:142] = True
    components, _ = ie.edge_components(strong)
    settings = {**ie.EDGE_DEFAULTS, "tiles": 4}

    scale_edges, _near, _info = ie.fit_prior_regions(prior, geometric, sparse, components, settings)
    scale_tiles, _near2, _info2 = ie.fit_prior_regions(prior, geometric, sparse, None, settings)

    niche = (slice(20, 220), slice(105, 135))
    err_edges = np.abs(prior[niche] * scale_edges[niche] / truth[niche] - 1)[scale_edges[niche] > 0]
    err_tiles = np.abs(prior[niche] * scale_tiles[niche] / truth[niche] - 1)[scale_tiles[niche] > 0]
    assert len(err_edges) > 0.8 * 200 * 30 and err_edges.mean() < 0.02
    assert err_tiles.mean() > 0.04  # the tile fit is dominated by the corridor wall and bleeds across the edge

    # without anchors of its own the niche must not borrow the corridor scale across the edge
    no_anchor = ie.fit_prior_regions(prior, geometric, None, components, settings)[0]
    assert np.all(no_anchor[niche] == 0) and (no_anchor[:, :90] > 0).mean() > 0.9


def test_sparse_corner_points_anchor_textureless_wall(tmp_path: Path) -> None:
    unit = 1.0
    rng = np.random.default_rng(7)
    views = _views(unit, count=3)
    depths, priors, anchors = [], {}, {}
    for view in views:
        depth, world, _ = _wall_with_niche(view, unit, niche=(0, 0, 0))
        depths.append(np.zeros_like(depth))  # no dense COLMAP depth at all
        priors[view.name] = (depth * 1.10 * (1 + rng.normal(0, 0.003, depth.shape))).astype(np.float32)
        corners = (np.abs(world[..., 0]) < 2.0) & (world[..., 2] > 0.0)
        rows, cols = np.nonzero(corners)
        pick = rng.choice(len(rows), 8, replace=False)  # a handful of reliable corner / edge points
        anchors[view.name] = (np.column_stack((cols[pick] + 0.5, rows[pick] + 0.5)).astype(np.float32),
                              depth[rows[pick], cols[pick]])
    stereo = _write_stereo(tmp_path, views, depths)
    with_sparse = PriorRefiner(views, stereo, priors.get, _settings(use_edges=False), anchors=anchors)
    without = PriorRefiner(views, stereo, priors.get, _settings(use_edges=False, use_sparse=False), anchors=anchors)

    view = views[1]
    truth, _w, _n = _wall_with_niche(view, unit, niche=(0, 0, 0))
    depth = with_sparse.depth(view.name)
    ok = depth > 0
    assert ok.mean() > 0.3
    assert np.abs(depth[ok] / truth[ok] - 1).mean() < 0.03  # 10 % bias removed by the corner points
    assert not np.any(without.depth(view.name) > 0)  # no anchors, no trusted prior


def test_da_jump_without_image_edge_rejected() -> None:
    h = w = 160
    prior = np.full((h, w), 3.0, np.float32)
    prior[:, 80:] = 3.9  # depth jump of 30 %
    geometric = np.zeros((h, w), np.float32)
    no_edge = {"strong": np.zeros((h, w), bool), "canny": np.zeros((h, w), bool), "lines": []}
    edge = np.zeros((h, w), bool)
    edge[:, 79:82] = True
    with_edge = {"strong": edge, "canny": edge.copy(), "lines": []}
    settings = {**ie.EDGE_DEFAULTS}

    halo, _wrong = ie.edge_consistency(prior, geometric, no_edge, settings)
    halo_edge, _ = ie.edge_consistency(prior, geometric, with_edge, settings)

    assert halo[:, 78:82].all()  # halo next to the unexplained jump
    assert not halo[:, :70].any() and not halo[:, 90:].any()
    assert not halo_edge.any()  # the same jump with a matching image edge is a real boundary

    # image edge with a COLMAP depth jump across it, prior stays smooth -> prior wrong there
    smooth = np.full((h, w), 3.0, np.float32)
    colmap = np.where(np.arange(w)[None, :] < 80, 3.0, 3.8).astype(np.float32) * np.ones((h, 1), np.float32)
    line = {"strong": edge, "canny": edge.copy(), "lines": [(80.0, 10.0, 80.0, 150.0, True)]}
    _h2, wrong = ie.edge_consistency(smooth, colmap, line, settings)
    assert wrong[40:120, 78:82].all() and not wrong[:, :60].any()


def test_textureless_ceiling_still_filled(tmp_path: Path) -> None:
    unit = 0.8157  # unscaled project: 1 m = 0.8157 units
    rng = np.random.default_rng(8)
    views = _views(unit, forward=(0.0, 1.0, 0.6))  # looking up at the ceiling at z = 2.5 m
    depths, priors, anchors, labels = [], {}, {}, {}
    for view in views:
        depth, world, = _hit(view, (0, 0, 1), -2.5 * unit)
        beams = (np.abs(world[..., 0] / unit) > 1.2) | (np.abs(world[..., 1] / unit - 2.5) > 1.4)  # measured around the white patch
        depths.append(np.where(beams, depth, 0.0).astype(np.float32))
        priors[view.name] = (depth * 1.01 * (1 + rng.normal(0, 0.003, depth.shape))).astype(np.float32)
        labels[view.name] = np.full((SIZE, SIZE), 3, np.uint8)
        anchors[view.name] = _anchors(view, depth, ~beams, 6, rng)
    stereo = _write_stereo(tmp_path, views, depths)
    settings = _settings(anchor_rule=True, units_per_metre=unit, max_incidence_deg=80.0)

    xyz, _rgb, cls, _votes, report = fill_from_depth_prior(
        views, stereo, priors.get, np.empty((0, 3)), settings, label_reader=labels.get, anchors=anchors)

    patch = (np.abs(xyz[:, 0] / unit) < 1.0) & (np.abs(xyz[:, 1] / unit - 2.5) < 1.2)
    assert np.count_nonzero(patch) > 300
    assert np.all(np.abs(xyz[patch][:, 2] / unit - 2.5) < 0.04)
    assert report["units_per_metre"] == pytest.approx(unit)


def test_units_per_metre_derived_from_storey_height() -> None:
    rng = np.random.default_rng(9)
    unit = 0.8157
    floor = rng.normal(0.31, 0.01, 4000)
    ceiling = rng.normal(0.31 + 2.5 * unit, 0.01, 4000)
    z = np.concatenate([floor, ceiling, rng.uniform(0.4, 2.2, 3000)])
    labels = np.concatenate([np.full(4000, 4), np.full(4000, 3), np.full(3000, 2)])
    ids = {"floor": 4, "ceiling": 3, "wall": 2}

    derived = derive_units_per_metre(z, labels, ids)

    assert derived["units_per_metre"] == pytest.approx(unit, abs=0.01) and derived["source"] == "storey_height"
    assert derive_units_per_metre(z[labels != 3], labels[labels != 3], ids)["units_per_metre"] == 1.0  # no ceiling: metric
    scaled = _scaled({**DEFAULTS, "units_per_metre": derived["units_per_metre"]})
    assert scaled["max_depth"] == pytest.approx(DEFAULTS["max_depth"] * unit, rel=0.01)
    assert scaled["min_baseline_m"] == pytest.approx(DEFAULTS["min_baseline_m"] * unit, rel=0.01)
    assert scaled["spacing_m"] == pytest.approx(DEFAULTS["spacing_m"] * unit, rel=0.01)
