"""Tests for samcloud.environment.ortho_rgb."""
import numpy as np

from samcloud.environment.ortho_rgb import OrthoRGB, build_ortho_rgb


def _cloud(seed=0):
    rng = np.random.default_rng(seed)
    n = 60000
    xy = rng.uniform(0.0, 10.0, (n, 2))
    z = rng.normal(0.0, 0.003, n)
    rgb = np.where((xy[:, 0] < 5.0)[:, None], np.array([200, 40, 40]), np.array([40, 40, 200])).astype(np.uint8)
    xyz = np.column_stack((xy, z))
    # a "roof" above x in [2, 3], y in [2, 3]: top-down colour must be the roof colour (green)
    roof = rng.uniform(2.0, 3.0, (4000, 2))
    xyz = np.vstack((xyz, np.column_stack((roof, rng.normal(5.0, 0.01, len(roof))))))
    rgb = np.vstack((rgb, np.tile(np.array([30, 220, 30], np.uint8), (len(roof), 1))))
    return xyz, rgb


def test_colours_at_known_positions():
    xyz, rgb = _cloud()
    ortho = build_ortho_rgb(xyz, rgb, cell_m=0.1, hole_fill_cells=0, bounds=(0.0, 0.0, 10.0, 10.0))
    assert ortho is not None and ortho.valid.mean() > 0.9
    lab_r, cov = ortho.sample_lab(np.array([[1.05, 7.05], [8.05, 7.05]]))
    assert cov.all()
    from skimage.color import rgb2lab
    want_r = rgb2lab(np.array([[[200, 40, 40]]], np.uint8) )[0, 0]
    want_b = rgb2lab(np.array([[[40, 40, 200]]], np.uint8))[0, 0]
    assert np.abs(lab_r[0] - want_r).max() < 2.0 and np.abs(lab_r[1] - want_b).max() < 2.0
    roof, _ = ortho.sample_lab(np.array([[2.5, 2.5]]))
    assert np.abs(roof[0] - rgb2lab(np.array([[[30, 220, 30]]], np.uint8))[0, 0]).max() < 2.0
    h, hc = ortho.sample_height(np.array([[2.5, 2.5], [8.05, 7.05]]))
    assert hc.all() and abs(h[0] - 5.0) < 0.05 and abs(h[1]) < 0.05


def test_holes_are_filled_and_big_gaps_stay():
    xyz, rgb = _cloud(1)
    keep = ~((xyz[:, 0] > 6.0) & (xyz[:, 0] < 6.5) & (xyz[:, 1] < 4.0))       # 50 cm wide strip
    keep &= ~((xyz[:, 0] > 8.0) & (xyz[:, 0] < 8.05) & (xyz[:, 1] < 4.0))      # 5 cm strip
    raw = build_ortho_rgb(xyz[keep], rgb[keep], cell_m=0.05, hole_fill_cells=0, bounds=(0, 0, 10, 10))
    filled = build_ortho_rgb(xyz[keep], rgb[keep], cell_m=0.05, hole_fill_cells=3, bounds=(0, 0, 10, 10))
    narrow = np.array([[8.025, 2.0]])
    wide = np.array([[6.25, 2.0]])
    assert not raw.sample_lab(narrow)[1].all()
    assert filled.sample_lab(narrow)[1].all()
    assert filled.valid.sum() > raw.valid.sum()
    assert not filled.sample_lab(wide)[1].all()      # 50 cm hole: the centre is 5 cells from valid data
    col, _ = filled.sample_lab(narrow)      # filled colour = red (x < 5) or blue (x >= 5) neighbours: blue here
    assert col[0][2] < -40.0


def test_no_colours_and_geotiff_roundtrip(tmp_path):
    xyz, rgb = _cloud(2)
    assert build_ortho_rgb(xyz, None) is None
    ortho = build_ortho_rgb(xyz, rgb, cell_m=0.1, bounds=(0, 0, 10, 10))
    path = ortho.to_geotiff(tmp_path / "ortho_rgb.tif")
    back = OrthoRGB.from_geotiff(path)
    assert back.shape == ortho.shape and np.array_equal(back.rgb, ortho.rgb) and np.array_equal(back.valid, ortho.valid)
    assert abs(back.xmin - ortho.xmin) < 1e-9 and abs(back.ymax - ortho.ymax) < 1e-9 and abs(back.cell - ortho.cell) < 1e-12
    ortho.preview_png(tmp_path / "p.png")
    assert (tmp_path / "p.png").is_file()
