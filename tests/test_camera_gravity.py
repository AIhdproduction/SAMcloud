"""Level floor for indoor camera captures: numeric core, model writer and pipeline gating."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

from samcloud.colmap import gravity
from samcloud.colmap.gravity import (
    GravityAlignmentError,
    align_camera_model,
    angle_degrees,
    camera_gravity_transform,
    estimate_camera_up,
    largest_horizontal_plane,
)
from samcloud.engine import pipeline

TILT_DEG = 12.0
Z = np.array([0.0, 0.0, 1.0])


def _tilt() -> np.ndarray:
    angle = math.radians(TILT_DEG)
    return np.array([
        [1.0, 0.0, 0.0],
        [0.0, math.cos(angle), -math.sin(angle)],
        [0.0, math.sin(angle), math.cos(angle)],
    ])


def _cam_from_world(yaw: float, pitch: float) -> np.ndarray:
    """World-to-camera rotation of an upright camera looking along yaw/pitch (COLMAP: x right, y down, z forward)."""
    forward = np.array([math.cos(pitch) * math.cos(yaw), math.cos(pitch) * math.sin(yaw), math.sin(pitch)])
    right = np.cross(forward, Z)
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    return np.vstack([right, down, forward])


def _scene(*, floor: int = 1200, ceiling: int = 0, walls: int = 600, seed: int = 7, noise: float = 0.004):
    """Room of 10 x 10 x 2.5 m in true coordinates, then tilted by 12 degrees into the 'model' frame."""
    rng = np.random.default_rng(seed)
    tilt = _tilt()
    parts = []
    if floor:
        parts.append(np.column_stack([rng.uniform(-5, 5, floor), rng.uniform(-5, 5, floor), np.zeros(floor)]))
    if ceiling:
        parts.append(np.column_stack([rng.uniform(-5, 5, ceiling), rng.uniform(-5, 5, ceiling), np.full(ceiling, 2.5)]))
    if walls:
        half = walls // 2
        parts.append(np.column_stack([np.full(half, 5.0), rng.uniform(-5, 5, half), rng.uniform(0, 2.5, half)]))
        parts.append(np.column_stack([rng.uniform(-5, 5, walls - half), np.full(walls - half, -5.0), rng.uniform(0, 2.5, walls - half)]))
    true_points = np.vstack(parts) + rng.normal(0.0, noise, (sum(len(p) for p in parts), 3))
    centres_true = np.column_stack([rng.uniform(-3, 3, 40), rng.uniform(-3, 3, 40), rng.normal(1.5, 0.05, 40)])
    rotations_true = [
        _cam_from_world(rng.uniform(-math.pi, math.pi), math.radians(rng.uniform(-15, 15))) for _ in range(40)
    ]
    points = true_points @ tilt.T
    centres = centres_true @ tilt.T
    rotations = [rotation @ tilt.T for rotation in rotations_true]  # cam_from_world of the tilted model
    return points, centres, rotations, tilt @ Z


# ------------------------------------------------------------------------------------------ numeric core


def test_tilted_floor_with_pitch_noise_is_levelled_within_half_a_degree() -> None:
    points, centres, rotations, true_up = _scene()
    estimate, translation, plane, floor_applied, extras = camera_gravity_transform(rotations, points, centres)

    assert extras["method"] == "camera_up_plane" and extras["warnings"] == []
    assert angle_degrees(estimate.up_world, true_up) < 0.5
    assert angle_degrees(estimate.rotation @ true_up, Z) < 0.5
    assert extras["tilt_corrected_deg"] == pytest.approx(TILT_DEG, abs=0.6)
    assert extras["image_count"] == 40 and extras["plane_inlier_count"] >= 800
    assert extras["coarse_to_final_angle_deg"] >= 0.0
    # the floor ends up horizontal and at z = 0 below the cameras
    assert plane is not None and plane.angle_to_z_deg < 0.5
    assert floor_applied is True
    assert abs(plane.z_at_origin + translation[2]) < 0.02


def test_coarse_camera_up_alone_is_close_but_not_exact() -> None:
    _points, _centres, rotations, true_up = _scene()
    coarse = estimate_camera_up(rotations)
    assert angle_degrees(coarse.up_world, true_up) < 6.0  # why the plane refinement exists
    assert coarse.input_count == 40 and coarse.retained_count >= 36


def test_no_plane_keeps_the_camera_up_and_warns() -> None:
    rng = np.random.default_rng(3)
    _points, centres, rotations, true_up = _scene()
    cloud = rng.uniform([-5, -5, 0], [5, 5, 3], (800, 3))  # volume, no dominant plane
    estimate, _t, _plane, _floor, extras = camera_gravity_transform(rotations, cloud, centres)
    assert extras["method"] == "camera_up_only" and extras["plane_inlier_count"] == 0
    assert extras["warnings"] and "refinement" in extras["warnings"][0]
    assert angle_degrees(estimate.up_world, estimate.rotation.T @ Z) < 1e-6
    assert angle_degrees(estimate.up_world, extras["coarse_up_world"]) < 1e-6

    _e, _t2, _p, _f, empty = camera_gravity_transform(rotations, np.zeros((0, 3)), centres)
    assert empty["method"] == "camera_up_only" and empty["warnings"]


def test_a_small_plane_below_the_threshold_is_not_used() -> None:
    points, centres, rotations, _up = _scene(floor=60, walls=0)
    _e, _t, _p, _f, extras = camera_gravity_transform(rotations, points, centres)
    assert extras["method"] == "camera_up_only"
    relaxed = camera_gravity_transform(rotations, points, centres, min_plane_inliers=50, min_plane_fraction=0.01)[4]
    assert relaxed["method"] == "camera_up_plane"


def test_ceiling_as_largest_plane_still_gives_the_right_up_sign() -> None:
    points, centres, rotations, true_up = _scene(floor=250, ceiling=1500)
    estimate, translation, _plane, floor_applied, extras = camera_gravity_transform(rotations, points, centres)

    assert extras["method"] == "camera_up_plane"
    assert float(estimate.up_world @ true_up) > 0.999 and angle_degrees(estimate.up_world, true_up) < 0.5
    assert estimate.rotation @ estimate.up_world == pytest.approx(Z, abs=1e-9)
    assert np.isfinite(translation).all() and isinstance(floor_applied, bool)


def test_largest_horizontal_plane_respects_a_custom_up() -> None:
    points, _centres, _rotations, true_up = _scene(walls=0)
    assert largest_horizontal_plane(points) is None or largest_horizontal_plane(points).angle_to_z_deg > 10.0
    plane = largest_horizontal_plane(points, max_angle_deg=20.0, up=true_up)
    assert plane is not None and plane.angle_to_z_deg < 0.5 and float(plane.normal @ true_up) > 0
    default = largest_horizontal_plane(np.column_stack([np.random.default_rng(1).uniform(-1, 1, (200, 2)), np.zeros(200)]))
    assert default is not None and default.angle_to_z_deg < 0.1  # panorama behaviour unchanged


def test_too_few_cameras_raise() -> None:
    with pytest.raises(GravityAlignmentError):
        estimate_camera_up([np.eye(3), np.eye(3)])
    with pytest.raises(GravityAlignmentError):
        estimate_camera_up([])


# ------------------------------------------------------------------------------------ model writer


def _write_model(directory: Path, points: np.ndarray, rotations: list[np.ndarray], centres: np.ndarray) -> None:
    import pycolmap

    reconstruction = pycolmap.Reconstruction()
    camera = pycolmap.Camera.create_from_model_name(1, "PINHOLE", 800.0, 640, 480)
    reconstruction.add_camera_with_trivial_rig(camera)
    for index, (rotation, centre) in enumerate(zip(rotations, centres), start=1):
        image = pycolmap.Image(name=f"frame_{index:06d}.jpg", camera_id=1, image_id=index)
        pose = pycolmap.Rigid3d(pycolmap.Rotation3d(rotation), -rotation @ centre)
        reconstruction.add_image_with_trivial_frame(image, pose)
    for point in points:
        reconstruction.add_point3D(point, pycolmap.Track())
    directory.mkdir(parents=True, exist_ok=True)
    reconstruction.write(directory)


def test_align_camera_model_writes_a_levelled_model_and_report(tmp_path: Path) -> None:
    import pycolmap

    points, centres, rotations, true_up = _scene()
    source = tmp_path / "sparse" / "0"
    _write_model(source, points, rotations, centres)

    report = align_camera_model(source, tmp_path / "sparse" / "0_gravity")

    assert report["method"] == "camera_up_plane" and report["image_count"] == 40
    assert report["tilt_corrected_deg"] == pytest.approx(TILT_DEG, abs=0.6)
    assert report["plane_inlier_count"] >= 800 and report["warnings"] == []
    assert angle_degrees(report["final_up_world"], true_up) < 0.5
    assert set(("rotation", "translation", "frame_count", "up_spread_deg_rms", "aligned_model")) <= set(report)
    json.dumps(report)  # JSON-ready
    aligned = pycolmap.Reconstruction(tmp_path / "sparse" / "0_gravity")
    aligned_points = np.asarray([point.xyz for point in aligned.points3D.values()])
    floor = largest_horizontal_plane(aligned_points)
    assert floor is not None and floor.angle_to_z_deg < 0.5
    ups = []
    for image in aligned.images.values():
        pose = image.cam_from_world() if callable(getattr(image, "cam_from_world", None)) else image.cam_from_world
        ups.append(pose.rotation.matrix().T @ gravity.CAMERA_UP_AXIS)
    assert angle_degrees(np.mean(ups, axis=0), Z) < 6.0


def test_align_camera_model_rejects_a_model_without_poses(tmp_path: Path) -> None:
    import pycolmap

    reconstruction = pycolmap.Reconstruction()
    (tmp_path / "empty").mkdir()
    reconstruction.write(tmp_path / "empty")
    with pytest.raises(GravityAlignmentError):
        align_camera_model(tmp_path / "empty", tmp_path / "out")


# --------------------------------------------------------------------------------------- pipeline gating


def _run_main(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, capture: str, class_set: str,
              tag_georeferenced: bool = False, switch: bool | None = None) -> tuple[list[Path], list[Path]]:
    from test_apriltag_pipeline import _stub_colmap

    images = tmp_path / "images"
    images.mkdir(exist_ok=True)
    (images / "image.jpg").write_bytes(b"image")
    camera_calls: list[Path] = []
    panorama_calls: list[Path] = []
    _stub_colmap(monkeypatch)
    monkeypatch.setattr(pipeline, "_apriltag_step", lambda model0, *_a, **_k: pipeline.TagStepResult(
        model0, {"mode": "georef" if tag_georeferenced else "scale"}, tag_georeferenced, False))
    monkeypatch.setattr(pipeline, "_align_camera_gravity", lambda sparse_dir, model0, report_dir: camera_calls.append(
        model0) or (model0, {"frame_count": 1, "up_spread_deg_rms": 0.0}))
    monkeypatch.setattr(pipeline, "_align_panorama_gravity", lambda sparse_dir, model0, report_dir: panorama_calls.append(
        model0) or (model0, {"frame_count": 1, "up_spread_deg_rms": 0.0}))
    if switch is not None:
        real_load = pipeline.load_global_config

        def load(name, *args, **kwargs):
            config = real_load(name, *args, **kwargs)
            if name == "colmap.yml":
                config = {**config, "alignment": {**config.get("alignment", {}), "gravity_align_camera_indoor": switch}}
            return config

        monkeypatch.setattr(pipeline, "load_global_config", load)
    argv = ["pipeline", "--capture-type", capture, "--class-set", class_set, "--outdir", str(tmp_path / "products"),
            "--apriltags", "--control-points-mode", "after_alignment", "--geometry-refinement", "0", "--skip-export"]
    if capture == "panorama_360":
        cubemap = tmp_path / "inputs" / "cubemap" / "front"
        cubemap.mkdir(parents=True, exist_ok=True)
        (cubemap / "pano_0001.jpg").write_bytes(b"image")
        monkeypatch.setattr(pipeline, "resolve_input_images", lambda *_args: (tmp_path / "inputs" / "cubemap", True, 256))
        argv += ["--panorama-workdir", str(tmp_path / "inputs")]
    else:
        argv += ["--images", str(images)]
    monkeypatch.setattr(sys, "argv", argv)
    pipeline.main()
    return camera_calls, panorama_calls


def test_indoor_camera_runs_the_camera_alignment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    camera, panorama = _run_main(tmp_path, monkeypatch, capture="camera", class_set="indoor")
    assert len(camera) == 1 and panorama == []
    final = (tmp_path / "products" / "sparse" / "final_model.txt").read_text(encoding="utf-8").strip()
    assert final == camera[0].name  # the stub keeps the model name, the real step selects 0_gravity


@pytest.mark.parametrize("capture,class_set", [("camera", "outdoor")])
def test_other_captures_skip_the_camera_alignment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capture: str, class_set: str,
) -> None:
    camera, panorama = _run_main(tmp_path, monkeypatch, capture=capture, class_set=class_set)
    assert camera == [] and panorama == []


def test_tag_georeferenced_indoor_camera_skips_the_alignment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    camera, _panorama = _run_main(tmp_path, monkeypatch, capture="camera", class_set="indoor", tag_georeferenced=True)
    assert camera == []


def test_config_switch_turns_the_camera_alignment_off(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    camera, _panorama = _run_main(tmp_path, monkeypatch, capture="camera", class_set="indoor", switch=False)
    assert camera == []


def test_panorama_keeps_its_own_alignment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    camera, panorama = _run_main(tmp_path, monkeypatch, capture="panorama_360", class_set="indoor")
    assert camera == [] and len(panorama) == 1


def test_a_failing_camera_alignment_never_stops_the_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def broken(*_args):
        raise GravityAlignmentError("not enough cameras")

    from test_apriltag_pipeline import _stub_colmap

    images = tmp_path / "images"
    images.mkdir()
    (images / "image.jpg").write_bytes(b"image")
    _stub_colmap(monkeypatch)
    monkeypatch.setattr(pipeline, "_align_camera_gravity", broken)
    monkeypatch.setattr(sys, "argv", [
        "pipeline", "--capture-type", "camera", "--class-set", "indoor", "--images", str(images),
        "--outdir", str(tmp_path / "products"), "--control-points-mode", "after_alignment",
        "--geometry-refinement", "0", "--skip-export",
    ])
    pipeline.main()
    assert (tmp_path / "products" / "sparse" / "final_model.txt").is_file()


def test_the_config_switch_exists_and_defaults_on() -> None:
    from samcloud.core.configuration import load_global_config

    alignment = load_global_config("colmap.yml")["alignment"]
    assert alignment["gravity_align_panorama"] is True and alignment["gravity_align_camera_indoor"] is True


# ------------------------------------------------------------- walls at camera height must not win (regression)


def _box_room(seed: int = 11):
    """5 x 4 x 2.5 m box room tilted by 12 degrees: many wall points, floor, ceiling, clutter, handheld cameras."""
    rng = np.random.default_rng(seed)
    tilt = _tilt()
    noise = 0.005
    walls = []
    for count, fixed_axis, value in ((9000, 0, 0.0), (9000, 0, 5.0), (6000, 1, 0.0), (6000, 1, 4.0)):
        other = 1 - fixed_axis
        span = 4.0 if fixed_axis == 0 else 5.0
        pts = np.zeros((count, 3))
        pts[:, fixed_axis] = value
        pts[:, other] = rng.uniform(0, span, count)
        pts[:, 2] = rng.uniform(0, 2.5, count)
        walls.append(pts)
    floor = np.column_stack([rng.uniform(0, 5, 2000), rng.uniform(0, 4, 2000), np.zeros(2000)])
    ceiling = np.column_stack([rng.uniform(0, 5, 1500), rng.uniform(0, 4, 1500), np.full(1500, 2.5)])
    clutter = np.column_stack([rng.uniform(0.2, 4.8, 2000), rng.uniform(0.2, 3.8, 2000), rng.uniform(0.7, 1.6, 2000)])
    true_points = np.vstack(walls + [floor, ceiling, clutter])
    true_points = true_points + rng.normal(0.0, noise, true_points.shape)
    centres_true = np.column_stack([rng.uniform(1, 4, 60), rng.uniform(1, 3, 60), rng.normal(1.4, 0.03, 60)])
    rotations = []
    for _ in range(60):
        base = _cam_from_world(rng.uniform(-math.pi, math.pi), math.radians(rng.uniform(-15, 15)))
        roll = math.radians(rng.uniform(-15, 15))
        about_forward = np.array([
            [math.cos(roll), -math.sin(roll), 0.0],
            [math.sin(roll), math.cos(roll), 0.0],
            [0.0, 0.0, 1.0],
        ])
        rotations.append(about_forward @ base @ tilt.T)
    return true_points @ tilt.T, centres_true @ tilt.T, rotations, tilt @ Z


def test_wall_heavy_room_with_clutter_is_levelled_by_the_floor_not_a_wall_band() -> None:
    points, centres, rotations, true_up = _box_room()
    coarse = estimate_camera_up(rotations)
    estimate, _translation, _plane, _floor, extras = camera_gravity_transform(rotations, points, centres)
    coarse_error = angle_degrees(coarse.up_world, true_up)
    final_error = angle_degrees(estimate.up_world, true_up)
    print(f"coarse error {coarse_error:.2f} deg, final error {final_error:.2f} deg, true tilt {TILT_DEG} deg")
    assert extras["method"] == "camera_up_plane"
    assert final_error < 2.0
    assert extras["horizontal_candidate_count"] > 0
    assert extras["plane_height_below_cameras"] is not None
    assert extras["plane_height_below_cameras"] == pytest.approx(1.4, abs=0.6) or extras["plane_height_below_cameras"] < 0


def test_a_plane_disagreeing_with_the_camera_axes_is_rejected() -> None:
    _points, centres, rotations, _true_up = _box_room()
    coarse = estimate_camera_up(rotations)
    axis = unit_vector_perp(coarse.up_world)
    slanted = rotation_about(axis, math.radians(15.0)) @ coarse.up_world
    rng = np.random.default_rng(5)
    basis_a = unit_vector_perp(slanted)
    basis_b = np.cross(slanted, basis_a)
    plane_points = (
        rng.uniform(-3, 3, (3000, 1)) * basis_a + rng.uniform(-3, 3, (3000, 1)) * basis_b
        + rng.normal(0.0, 0.004, (3000, 3))
    )
    estimate, _t, _plane, _floor, extras = camera_gravity_transform(rotations, plane_points, centres)
    assert extras["method"] == "camera_up_only" and extras["warnings"]
    assert angle_degrees(estimate.up_world, coarse.up_world) < 1e-6


def test_refined_normal_beyond_the_limit_is_rejected_with_a_warning(monkeypatch: pytest.MonkeyPatch) -> None:
    points, centres, rotations, _true_up = _box_room()
    coarse = estimate_camera_up(rotations)
    tilted = rotation_about(unit_vector_perp(coarse.up_world), math.radians(15.0)) @ coarse.up_world
    fake = gravity.PlaneFit(normal=tilted, offset=0.0, inlier_count=5000, point_count=6000, angle_to_z_deg=15.0)
    monkeypatch.setattr(gravity, "largest_horizontal_plane", lambda *a, **k: fake)
    estimate, _t, _plane, _floor, extras = camera_gravity_transform(rotations, points, centres)
    assert extras["method"] == "camera_up_only"
    assert any("disagreed with the camera axes by 15.0 degrees" in w for w in extras["warnings"])
    assert angle_degrees(estimate.up_world, coarse.up_world) < 1e-6


def unit_vector_perp(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=float)
    helper = np.eye(3)[int(np.argmin(np.abs(vector)))]
    result = np.cross(vector, helper)
    return result / np.linalg.norm(result)


def rotation_about(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=float) / np.linalg.norm(axis)
    skew = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + math.sin(angle) * skew + (1 - math.cos(angle)) * skew @ skew


def test_normal_filter_keeps_floor_points_and_drops_wall_points() -> None:
    rng = np.random.default_rng(2)
    floor = np.column_stack([rng.uniform(0, 5, 1500), rng.uniform(0, 4, 1500), rng.normal(0, 0.003, 1500)])
    wall = np.column_stack([rng.normal(0, 0.003, 1500), rng.uniform(0, 4, 1500), rng.uniform(0.5, 2.5, 1500)])
    kept = gravity._horizontal_surface_points(np.vstack([floor, wall]), Z)
    assert 1200 <= len(kept)
    assert float(np.mean(kept[:, 2] < 0.05)) > 0.98  # only floor-level points survive
    assert len(gravity._horizontal_surface_points(np.zeros((2, 3)), Z)) == 0


def test_scale_field_reports_apriltag_for_tag_scaled_sources(tmp_path: Path) -> None:
    points, centres, rotations, _up = _scene()
    plain = tmp_path / "sparse" / "0"
    tagged = tmp_path / "sparse" / "0_tagscale"
    _write_model(plain, points, rotations, centres)
    _write_model(tagged, points, rotations, centres)
    assert align_camera_model(plain, tmp_path / "a")["scale"] == "arbitrary (no reference)"
    assert align_camera_model(tagged, tmp_path / "b")["scale"] == "apriltag"
