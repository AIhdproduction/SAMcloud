"""Tests for samcloud.drone.quality (synthetic data only)."""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np

from samcloud.drone import quality as q


def _recon(errors, tracks, registered):
    points = {i: SimpleNamespace(error=e, track=SimpleNamespace(length=lambda t=t: t))
              for i, (e, t) in enumerate(zip(errors, tracks))}
    return SimpleNamespace(reg_image_ids=lambda: list(range(registered)), points3D=points)


def test_sparse_metrics_registered_ratio_and_stats():
    rng = np.random.default_rng(1)
    errors = rng.uniform(0.2, 1.1, 501)
    tracks = rng.integers(2, 12, 501)
    block = q.compute_sparse_metrics(_recon(errors, tracks, 87), 100)
    assert block["registered_ratio"] == 0.87
    assert block["num_points3D"] == 501
    assert block["num_points3d"] == 501
    assert abs(block["reprojection_error_mean_px"] - errors.mean()) < 1e-9
    assert abs(block["reprojection_error_median_px"] - np.median(errors)) < 1e-9
    assert abs(block["track_length_mean"] - tracks.mean()) < 1e-9
    assert block["track_length_median"] == float(np.median(tracks))


def test_umeyama_recovers_noise_rms():
    rng = np.random.default_rng(2)
    n, sigma = 400, 0.03
    true = rng.uniform(-50, 50, (n, 3))
    angle = 0.7
    rot = np.array([[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1.0]])
    model = (true - 5.0) / 1.7 @ rot.T + np.array([3.0, -2.0, 1.0])  # unknown frame
    noise = rng.normal(0, sigma, (n, 3))
    names = [f"img{i}.jpg" for i in range(n)]
    block = q.compute_rtk_residuals(dict(zip(names, model)), dict(zip(names, true + noise)))
    assert block["available"] and block["matched_images"] == n
    for axis in "ENU":
        assert abs(block["after_similarity"][axis]["rms"] - sigma) < 0.1 * sigma
    # frame offset makes the raw residual much larger than the fitted one
    assert block["raw"]["E"]["rms"] > 10 * sigma
    assert abs(block["after_similarity"]["scale"] - 1.7) < 0.02


def test_rtk_geodetic_reference_is_converted_to_enu():
    rng = np.random.default_rng(3)
    n = 60
    enu = rng.uniform(-40, 40, (n, 3))
    lat0, lon0 = 47.05, 8.30
    llh = np.column_stack((lat0 + enu[:, 1] / 111_200.0, lon0 + enu[:, 0] / (111_320.0 * np.cos(np.radians(lat0))),
                           400.0 + enu[:, 2]))
    names = [f"i{i}" for i in range(n)]
    block = q.compute_rtk_residuals(dict(zip(names, enu)), dict(zip(names, llh)))
    assert block["reference_frame"] == "wgs84_local_enu"
    assert block["after_similarity"]["rms_3d"] < 0.5  # linear approx of lat/lon is metre-accurate here


def test_plane_noise_known_sigma():
    rng = np.random.default_rng(4)
    n = 150_000
    xyz = np.column_stack((rng.uniform(0, 10, n), rng.uniform(0, 10, n), rng.normal(0, 0.01, n)))
    result = q.compute_plane_noise(xyz, np.full(n, 11, dtype=np.uint16))
    road = result["classes"]["road"]
    assert road["patches"] >= 20
    assert 0.008 <= road["rms_m_median"] <= 0.012
    assert result["classes"]["ground"]["available"] is False


def test_density_and_gap_share_with_hole():
    rng = np.random.default_rng(5)
    pts = rng.uniform(0.01, 39.99, (200_000, 2))
    inside = (pts[:, 0] > 15.2) & (pts[:, 0] < 24.8) & (pts[:, 1] > 15.2) & (pts[:, 1] < 24.8)
    xyz = np.column_stack((pts[~inside], rng.normal(0, 0.01, int((~inside).sum()))))
    block = q.compute_density(xyz)
    n_kept = len(xyz)
    expected = n_kept / (1600 - 9 ** 2 * 1.0)  # ~ hole of 9x9 cells fully empty (cells 16..24)
    assert 0.8 * expected < block["points_per_m2_median"] < 1.2 * expected
    hole_cells = 81
    assert abs(block["gap_share"] - hole_cells / 1600) < 0.02
    full = q.compute_density(np.column_stack((pts, np.zeros(len(pts)))))
    assert full["gap_share"] < 0.005


def test_facade_share_on_box_building():
    rng = np.random.default_rng(6)
    n_wall, n_roof = 60_000, 20_000
    w, d, h = 10.0, 8.0, 6.0
    u = rng.uniform(0, 1, n_wall)
    z = rng.uniform(0, h, n_wall)
    side = rng.integers(0, 4, n_wall)
    x = np.where(side == 0, 0, np.where(side == 1, w, u * w))
    y = np.where(side == 2, 0, np.where(side == 3, d, u * d))
    x = np.where(side < 2, x, u * w)
    y = np.where(side < 2, u * d, y)
    walls = np.column_stack((x, y, z))
    roof = np.column_stack((rng.uniform(0, w, n_roof), rng.uniform(0, d, n_roof), np.full(n_roof, h)))
    xyz = np.vstack((walls, roof)) + rng.normal(0, 0.003, (n_wall + n_roof, 3))
    block = q.compute_facade_coverage(xyz, np.full(len(xyz), 6, dtype=np.uint16))
    assert abs(block["facade_share"] - 0.75) < 0.05
    assert block["roof_share"] > 0.15


def test_missing_inputs_give_null_and_reason(tmp_path):
    metrics = q.compute_quality(tmp_path, run_name="empty")
    for key in ("sparse", "rtk", "density", "plane_noise", "facade_coverage"):
        assert metrics[key]["available"] is False
        assert metrics[key]["reason"]
    path = q.write_quality_report(metrics, tmp_path / "out")
    assert path.name == "quality_empty.json"
    assert json.loads(path.read_text(encoding="utf-8"))["run"] == "empty"
    assert "n/a" in q.compare_runs([metrics])


def _write_ply(path, xyz, cls):
    header = ("ply\nformat binary_little_endian 1.0\nelement vertex %d\nproperty float x\nproperty float y\n"
              "property float z\nproperty uchar scalar_class\nend_header\n" % len(xyz))
    data = np.zeros(len(xyz), dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("c", "u1")])
    data["x"], data["y"], data["z"], data["c"] = xyz[:, 0], xyz[:, 1], xyz[:, 2], cls
    with open(path, "wb") as fh:
        fh.write(header.encode("ascii"))
        fh.write(data.tobytes())


def test_compute_quality_end_to_end_with_ply(tmp_path):
    rng = np.random.default_rng(7)
    n = 60_000
    xyz = np.column_stack((rng.uniform(0, 10, n), rng.uniform(0, 10, n), rng.normal(0, 0.01, n)))
    ply = tmp_path / "classified" / "classified.ply"
    ply.parent.mkdir()
    _write_ply(ply, xyz, np.full(n, 2, dtype=np.uint8))
    metrics = q.compute_quality(tmp_path, run_name="e2e")
    assert metrics["density"]["available"] and metrics["density"]["source"] == "classified.ply"
    assert 0.008 <= metrics["plane_noise"]["classes"]["road"]["rms_m_median"] <= 0.012
    assert metrics["sparse"]["available"] is False


def test_compare_runs_table_has_both_run_names():
    a = {"run": "baseline", "sparse": {"available": True, "registered_ratio": 0.8}}
    b = {"run": "rtk_opt", "sparse": {"available": True, "registered_ratio": 0.95},
         "density": {"available": True, "points_per_m2_median": 120.0}}
    table = q.compare_runs([a, b])
    header = table.splitlines()[0]
    assert "baseline" in header and "rtk_opt" in header
    row = next(line for line in table.splitlines() if line.startswith("| Registered ratio"))
    assert "0.800" in row and "0.950" in row
    assert "n/a" in next(line for line in table.splitlines() if line.startswith("| Density median"))


def test_rtk_raw_uses_first_reference_row_as_enu_origin():
    rng = np.random.default_rng(7)
    n = 40
    enu = rng.uniform(-60, 60, (n, 3))
    enu[0] = (25.0, -30.0, 2.0)  # first gps_ref row, deliberately away from the centroid
    lat0, lon0, alt0 = 47.05, 8.30, 400.0
    llh = np.column_stack((lat0 + enu[:, 1] / 111_200.0, lon0 + enu[:, 0] / (111_320.0 * np.cos(np.radians(lat0))),
                           alt0 + enu[:, 2]))
    names = [f"i{i:02d}" for i in range(n)]
    # model frame = ENU relative to the first row (what COLMAP model_aligner produces)
    model = enu - enu[0]
    block = q.compute_rtk_residuals(dict(zip(names, model)), dict(zip(names, llh)))
    for axis in "ENU":
        assert block["raw"][axis]["rms"] < 0.5, block["raw"]
    # a centroid-based origin would show a large offset (several tens of metres)
    centroid_ref = q.geodetic_to_local_enu(llh)
    assert np.abs(model - centroid_ref).mean() > 5.0
