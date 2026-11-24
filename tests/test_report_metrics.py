"""Measurement functions of the processing protocol (samcloud.report.metrics) on synthetic data."""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import numpy as np
import pycolmap
import pytest

from samcloud.report import metrics

FOCAL, CENTRE, DEPTH = 100.0, 50.0, 5.0


def build_model(images: dict[int, tuple[str, float]], points: list[tuple[tuple[float, float, float], list[int]]],
                offsets: dict[int, tuple[float, float]] | None = None) -> "pycolmap.Reconstruction":
    """Tiny reconstruction: pinhole cameras looking along +z at x-centres ``images[id] = (name, cx)``.

    Observations are exact projections plus an optional constant pixel offset per image.
    """
    offsets = offsets or {}
    model = pycolmap.Reconstruction()
    model.add_camera_with_trivial_rig(
        pycolmap.Camera(camera_id=1, model="PINHOLE", width=100, height=100, params=[FOCAL, FOCAL, CENTRE, CENTRE]))
    seen: dict[int, list[int]] = {image_id: [] for image_id in images}
    for index, (_, viewers) in enumerate(points):
        for image_id in viewers:
            seen[image_id].append(index)
    for image_id, (name, cx) in images.items():
        image = pycolmap.Image(name=name, camera_id=1, image_id=image_id)
        du, dv = offsets.get(image_id, (0.0, 0.0))
        xys = []
        for index in seen[image_id]:
            x, y, z = points[index][0]
            xys.append(pycolmap.Point2D([FOCAL * (x - cx) / z + CENTRE + du, FOCAL * y / z + CENTRE + dv]))
        image.points2D = pycolmap.Point2DList(xys)
        model.add_image_with_trivial_frame(
            image, pycolmap.Rigid3d(pycolmap.Rotation3d(), np.array([-cx, 0.0, 0.0])))
    for index, (xyz, viewers) in enumerate(points):
        track = pycolmap.Track()
        for image_id in viewers:
            track.add_element(image_id, seen[image_id].index(index))
        model.add_point3D(np.array(xyz, dtype=float), track)
    return model


def two_cluster_points(count: int = 200, seed: int = 7):
    rng = np.random.default_rng(seed)
    points = []
    for _ in range(count):  # left cluster x in [0.1, 1.9): seen by images 1 and 2 only
        points.append(((rng.uniform(0.1, 1.9), rng.uniform(0.1, 1.9), DEPTH), [1, 2]))
    for _ in range(count):  # right cluster x in [8.1, 9.9): seen by images 2, 3 and 4
        points.append(((rng.uniform(8.1, 9.9), rng.uniform(0.1, 1.9), DEPTH), [2, 3, 4]))
    return points


IMAGES = {1: ("a.jpg", 0.0), 2: ("b.jpg", 1.0), 3: ("c.jpg", 3.0), 4: ("d.jpg", 4.0)}


def test_alignment_metrics_counts_unregistered_and_errors(tmp_path: Path) -> None:
    model = build_model(IMAGES, two_cluster_points(), offsets={1: (3.0, 4.0)})  # image 1: exactly 5 px off
    for name in ("0", "1"):
        folder = tmp_path / "sparse" / name
        folder.mkdir(parents=True)
        model.write(str(folder))
    names = ["a.jpg", "b.jpg", "c.jpg", "d.jpg", "e.jpg", "sub/f.jpg"]
    result = metrics.alignment_metrics(tmp_path / "sparse" / "0", names)
    assert result["registered_images"] == 4
    assert result["total_images"] == 6
    assert result["unregistered_images"] == ["e.jpg", "sub/f.jpg"]
    assert result["submodels"] == 2
    assert result["tie_points"] == 400
    assert result["mean_track_length"] == pytest.approx((200 * 2 + 200 * 3) / 400)
    assert result["cameras"][0]["model"] == "PINHOLE"
    assert result["cameras"][0]["params"]["fx"] == pytest.approx(FOCAL)
    assert result["cameras"][0]["params"]["focal_px"] == pytest.approx(FOCAL)
    by_name = {item["name"]: item for item in result["images"]}
    assert by_name["a.jpg"]["reprojection_rms_px"] == pytest.approx(5.0, abs=1e-3)
    assert by_name["b.jpg"]["reprojection_rms_px"] == pytest.approx(0.0, abs=1e-3)
    assert by_name["a.jpg"]["observations"] == 200
    assert by_name["b.jpg"]["observations"] == 400
    assert by_name["c.jpg"]["center_xyz"] == pytest.approx([3.0, 0.0, 0.0], abs=1e-6)


def test_alignment_metrics_accepts_reconstruction_and_unknown_inputs() -> None:
    result = metrics.alignment_metrics(build_model(IMAGES, two_cluster_points(20)))
    assert result["registered_images"] == 4
    assert result["total_images"] is None and result["unregistered_images"] == [] and result["submodels"] is None


def test_overlap_grid_counts_distinct_images() -> None:
    grid = metrics.overlap_grid(build_model(IMAGES, two_cluster_points()), max_cells=10)
    counts = np.array(grid["counts"])
    assert list(counts.shape) == grid["shape"]
    assert set(np.unique(counts)) <= {0, 2, 3}
    assert grid["max"] == 3
    columns = np.where(counts.max(axis=0) > 0)[0]
    assert counts[:, columns[0]].max() == 2  # leftmost occupied column: images 1 and 2
    assert counts[:, columns[-1]].max() == 3  # rightmost: images 2, 3 and 4
    assert 2.0 <= grid["mean_over_occupied"] <= 3.0
    assert grid["cell_size"] == pytest.approx((np.ptp([p[0][0] for p in two_cluster_points()]) / 10), rel=0.1)


def test_overlap_grid_samples_deterministically() -> None:
    model = build_model(IMAGES, two_cluster_points())
    first = metrics.overlap_grid(model, max_cells=10, max_points=100)
    second = metrics.overlap_grid(model, max_cells=10, max_points=100)
    assert first["sampled_points"] == 100
    assert first["counts"] == second["counts"]


def test_confidence_histogram_and_percentiles(tmp_path: Path) -> None:
    classes = np.array([1, 1, 1, 1, 2, 2, 0, 0], dtype=np.uint8)
    conf = np.array([0.05, 0.15, 0.25, 0.95, 0.55, 0.65, 0.0, 0.0], dtype=np.float16)
    path = tmp_path / "classified_scores.npz"
    np.savez(path, top1_class=classes, top1_conf=conf)
    result = metrics.confidence_histogram(path, bins=10, class_names=["wall", "floor"])
    assert result["counts"] == [1, 1, 1, 0, 0, 1, 1, 0, 0, 1]
    assert result["points"] == 6 and result["unknown_points"] == 2
    assert result["per_class"]["wall"]["points"] == 4
    assert result["per_class"]["floor"]["p50"] == pytest.approx(0.6, abs=0.01)
    assert len(result["edges"]) == 11
    # the gzip variant next to a missing npz is read as well
    zipped = tmp_path / "gz" / "classified_scores.npz"
    zipped.parent.mkdir()
    with gzip.open(str(zipped) + ".gz", "wb") as target:
        target.write(path.read_bytes())
    assert metrics.confidence_histogram(zipped, bins=10)["counts"] == result["counts"]


def _expected_cells(xy: np.ndarray, cell: float) -> int:
    idx = np.floor((xy - xy.min(axis=0)) / cell).astype(np.int64)
    return len({(int(a), int(b)) for a, b in idx})


def test_dense_density_metric_and_not_metric(tmp_path: Path) -> None:
    rng = np.random.default_rng(3)
    xy = np.column_stack([rng.uniform(0.0, 9.93, 3000), rng.uniform(0.0, 7.41, 3000)])
    result = metrics.dense_density(xy, metric=True, cell_m=1.0)
    cells = _expected_cells(xy, 1.0)
    assert result["points"] == 3000
    assert result["area_m2"] == pytest.approx(cells)
    assert result["density_pts_m2"] == pytest.approx(3000 / cells)
    unscaled = metrics.dense_density(xy, metric=False, cell_m=1.0)
    assert unscaled["density_pts_m2"] is None and unscaled["reason"] == "no metric scale"
    assert unscaled["points"] == 3000
    # the same through a binary PLY (x/y read via memory map)
    vertex = np.zeros(3000, dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("red", "u1")])
    vertex["x"], vertex["y"] = xy[:, 0], xy[:, 1]
    ply = tmp_path / "cloud.ply"
    header = ("ply\nformat binary_little_endian 1.0\nelement vertex 3000\nproperty float x\nproperty float y\n"
              "property float z\nproperty uchar red\nend_header\n").encode("ascii")
    ply.write_bytes(header + vertex.tobytes())
    from_ply = metrics.dense_density(ply, metric=True, cell_m=1.0)
    assert from_ply["points"] == 3000
    assert from_ply["area_m2"] == pytest.approx(result["area_m2"], abs=3.0)  # float32 rounding at cell borders
    assert metrics.ply_vertex_count(ply) == 3000
    packed = tmp_path / "cloud_gz" / "fused.ply.gz"
    packed.parent.mkdir()
    with gzip.open(packed, "wb") as target:
        target.write(header + vertex.tobytes())
    assert metrics.ply_vertex_count(packed) == 3000


def test_refinement_stages_from_synthetic_run_metrics() -> None:
    run_metrics = {
        "settings": {
            "depth_prior": {"views": 10},
            "depth_fill": {"enabled": True, "original_points": 1000, "added_points": 200},
            "point_cleaning": {"error": "boom"},
            "surface_fusion": {"enabled": True, "original_points": 1200, "final_points": 900, "mirrors": [{}, {}],
                               "mirror_points_removed": 30, "removed_visibility": 50, "removed_hidden": 7},
            "plane_fill": {"enabled": True, "added_points": 5},
            "superpoint_vote": {"point_count": 900, "changed_points": {"count": 12}},
        },
        "stage_seconds": {"Depth fill": 12.5, "Surface fusion": 99.0, "Point cleaning": 1.0},
    }
    stages = metrics.refinement_stages(run_metrics, ["WARNING: Plane filling was skipped: no planes"])
    by_name = {row["stage"]: row for row in stages}
    order = [row["stage"] for row in stages]
    assert order[:5] == ["depth_prior", "mirror_detection", "depth_fill", "point_cleaning", "surface_fusion"]
    assert order.index("surface_fusion.removed_visibility") > order.index("surface_fusion")
    assert order[-3:] == ["plane_fill", "superpoint_vote", "depth_evidence"]
    assert by_name["depth_fill"]["after"] == 1200 and by_name["depth_fill"]["added"] == 200
    assert by_name["depth_fill"]["seconds"] == 12.5
    assert by_name["point_cleaning"]["skipped"] and by_name["point_cleaning"]["reason"] == "boom"
    assert by_name["surface_fusion"]["removed"] == 300 and by_name["surface_fusion"]["after"] == 900
    assert by_name["surface_fusion.removed_hidden"]["removed"] == 7
    assert by_name["mirror_detection"]["removed"] == 30 and by_name["mirror_detection"]["detected"] == 2
    assert by_name["plane_fill"]["skipped"] and "no planes" in by_name["plane_fill"]["reason"]
    assert by_name["depth_evidence"]["skipped"] and by_name["depth_evidence"]["reason"] == "not run"
    assert by_name["superpoint_vote"]["changed"] == 12
    assert metrics.refinement_stages({})[0]["skipped"] is True  # nothing recorded: every stage skipped, no exception


def test_log_warnings_dedupes_and_keeps_order(tmp_path: Path) -> None:
    log = tmp_path / "pipeline.log"
    log.write_text("start\nWARNING: a\nok\nWARNING: b\nWARNING: a\nWARNING [x] c\n", encoding="utf-8")
    assert metrics.log_warnings(log) == ["WARNING: a", "WARNING: b", "WARNING [x] c"]
    assert metrics.log_warnings(log, limit=1) == ["WARNING: a"]
    assert metrics.log_warnings(tmp_path / "missing.log") == []


def test_software_and_hardware_info_have_documented_keys() -> None:
    software = metrics.software_info()
    assert set(software) == {"version", "git_hash", "git_dirty"} and software["version"]
    hardware = metrics.hardware_info()
    assert {"gpu_name", "vram_gb", "ram_gb", "cpu", "cpu_count", "os"} <= set(hardware)
    assert hardware["cpu_count"] and hardware["os"]
    json.dumps(software)
    json.dumps(hardware)
