"""Tests for AprilTag triangulation and scale recovery on a synthetic room."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pycolmap
import pytest

from samcloud.control_points.apriltags import (
    APRILTAG_SIZE_M,
    TagObservation,
    apply_scale,
    build_tag_report,
    expected_edge_lengths,
    save_tag_report,
    tag_edge_lengths,
    tag_scale,
    triangulate_tags,
)

TRUE_SCALE = 2.37
WIDTH, HEIGHT, FOCAL = 1920, 1440, 1000.0

# (centre, inward normal) of tags on four different walls of a 5 x 4 x 2.6 m room.
TAG_WALLS = {
    11: (np.array([0.0, 1.71, 1.43]), np.array([1.0, 0.0, 0.0])),
    12: (np.array([5.0, 2.37, 1.27]), np.array([-1.0, 0.0, 0.0])),
    13: (np.array([2.23, 0.0, 1.52]), np.array([0.0, 1.0, 0.0])),
    14: (np.array([3.11, 4.0, 1.19]), np.array([0.0, -1.0, 0.0])),
}


def _tag_corners(tag_id: int, size: float, rotation_deg: float) -> np.ndarray:
    centre, normal = TAG_WALLS[tag_id]
    in_plane = np.cross(normal, [0.0, 0.0, 1.0])
    in_plane /= np.linalg.norm(in_plane)
    up = np.array([0.0, 0.0, 1.0])
    angle = np.deg2rad(rotation_deg)
    u = np.cos(angle) * in_plane + np.sin(angle) * up
    v = -np.sin(angle) * in_plane + np.cos(angle) * up
    half = size / 2.0
    return np.array([centre - half * u + half * v, centre + half * u + half * v,
                     centre + half * u - half * v, centre - half * u - half * v])


def _look_at(centre: np.ndarray, target: np.ndarray, roll_deg: float) -> np.ndarray:
    forward = (target - centre) / np.linalg.norm(target - centre)
    right = np.cross(forward, [0.0, 0.0, 1.0])
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    roll = np.deg2rad(roll_deg)
    right, down = np.cos(roll) * right + np.sin(roll) * down, -np.sin(roll) * right + np.cos(roll) * down
    return np.vstack((right, down, forward))


def _camera_poses() -> list[tuple[np.ndarray, np.ndarray]]:
    """Two oblique cameras per tag wall, with pitch and roll offsets."""
    poses = []
    rng = np.random.default_rng(11)
    for tag_id, (centre, normal) in TAG_WALLS.items():
        lateral = np.cross(normal, [0.0, 0.0, 1.0])
        for side, sign in enumerate((-1.0, 1.0)):
            position = centre + 2.2 * normal + sign * (0.9 + 0.13 * side) * lateral
            position[2] = 1.35 + rng.uniform(-0.15, 0.15)
            target = centre + rng.uniform(-0.3, 0.3, 3)
            poses.append((position, _look_at(position, target, rng.uniform(-6.0, 6.0))))
    return poses


def _build_reconstruction(root: Path) -> tuple[pycolmap.Reconstruction, Path]:
    """Room model in a coordinate frame that is 1/TRUE_SCALE of metric size."""
    reconstruction = pycolmap.Reconstruction()
    camera = pycolmap.Camera(camera_id=1, model="PINHOLE", width=WIDTH, height=HEIGHT,
                             params=np.array([FOCAL, FOCAL * 1.004, WIDTH / 2 + 3.7, HEIGHT / 2 - 2.9]))
    reconstruction.add_camera_with_trivial_rig(camera)
    poses = _camera_poses()
    for index, (position, rotation) in enumerate(poses, start=1):
        shrunk = position / TRUE_SCALE
        pose = pycolmap.Rigid3d(pycolmap.Rotation3d(rotation), -rotation @ shrunk)
        image = pycolmap.Image(name=f"room_{index:02d}.jpg", camera_id=1, image_id=index)
        reconstruction.add_image_with_trivial_frame(image, pose)
    rng = np.random.default_rng(23)
    room = np.array([5.0, 4.0, 2.6])
    for point in rng.uniform([0.4, 0.4, 0.3], room - [0.4, 0.4, 0.3], (60, 3)):
        track = pycolmap.Track()
        for image_id in reconstruction.reg_image_ids():
            image = reconstruction.image(image_id)
            uv = _project(reconstruction, image_id, point / TRUE_SCALE)
            if uv is not None:
                image.points2D.append(pycolmap.Point2D(uv))
                track.add_element(image_id, image.num_points2D() - 1)
        if track.length() >= 2:
            reconstruction.add_point3D(point / TRUE_SCALE, track)
    assert reconstruction.num_points3D() >= 10
    model_dir = root / "model"
    model_dir.mkdir(parents=True, exist_ok=True)
    reconstruction.write(model_dir)
    return reconstruction, model_dir


def _project(reconstruction: pycolmap.Reconstruction, image_id: int, xyz: np.ndarray) -> np.ndarray | None:
    image = reconstruction.image(image_id)
    camera = reconstruction.camera(image.camera_id)
    cam_point = np.asarray(image.cam_from_world() * xyz)
    if cam_point[2] <= 0:
        return None
    uv = np.asarray(camera.img_from_cam(cam_point))
    if not (5 < uv[0] < WIDTH - 5 and 5 < uv[1] < HEIGHT - 5):
        return None
    return uv


def _observations(reconstruction: pycolmap.Reconstruction, tag_sizes: dict[int, float], seed: int = 31) -> dict[str, list[TagObservation]]:
    rng = np.random.default_rng(seed)
    detections: dict[str, list[TagObservation]] = {}
    for tag_id, size in tag_sizes.items():
        corners = _tag_corners(tag_id, size, rotation_deg=float(rng.uniform(-80, 80))) / TRUE_SCALE
        centre = corners.mean(axis=0)
        normal = TAG_WALLS[tag_id][1]
        for image_id in reconstruction.reg_image_ids():
            image = reconstruction.image(image_id)
            camera_centre = np.asarray(image.cam_from_world().inverse().translation)
            if float(np.dot(camera_centre - centre, normal)) <= 0.2:
                continue
            projected = [_project(reconstruction, image_id, corner) for corner in corners]
            if any(p is None for p in projected):
                continue
            noisy = [p + rng.normal(0.0, 0.2, 2) for p in projected]
            # Centre from the noisy diagonals, as the detector would report it.
            p0, p1, p2, p3 = noisy
            t = np.linalg.solve(np.array([p2 - p0, -(p3 - p1)]).T, p1 - p0)[0]
            centre_uv = p0 + t * (p2 - p0)
            detections.setdefault(image.name, []).append(TagObservation(
                image=image.name, tag_id=tag_id,
                corners=tuple((float(x), float(y)) for x, y in noisy),
                center=(float(centre_uv[0]), float(centre_uv[1])),
            ))
    seen = {tag_id: sum(1 for obs in detections.values() for o in obs if o.tag_id == tag_id) for tag_id in tag_sizes}
    assert all(count >= 2 for count in seen.values()), seen
    return detections


@pytest.fixture
def room(tmp_path: Path) -> tuple[pycolmap.Reconstruction, Path]:
    return _build_reconstruction(tmp_path)


def test_three_tags_recover_scale_without_warning(room) -> None:
    reconstruction, _ = room
    detections = _observations(reconstruction, {11: 0.150, 12: 0.150, 13: 0.150})

    geometries = triangulate_tags(reconstruction, detections)
    result = tag_scale(geometries)

    assert sorted(geometries) == [11, 12, 13]
    assert result.scale_factor == pytest.approx(TRUE_SCALE, rel=0.005)
    assert not any("keine Kontrolle" in w for w in result.warnings)
    assert result.outliers == ()
    assert all(geometries[t].reprojection_rms_px < 1.0 for t in geometries)
    assert all(info["status"] == "ok" for info in result.per_tag.values())


def test_single_tag_is_scaled_with_warning(room) -> None:
    reconstruction, _ = room
    detections = _observations(reconstruction, {13: 0.150})

    result = tag_scale(triangulate_tags(reconstruction, detections))

    assert result.scale_factor == pytest.approx(TRUE_SCALE, rel=0.01)
    assert any("keine Kontrolle möglich" in w for w in result.warnings)


def test_no_tags_leaves_model_unscaled(room) -> None:
    reconstruction, _ = room

    geometries = triangulate_tags(reconstruction, {})
    result = tag_scale(geometries)

    assert geometries == {}
    assert result.scale_factor is None
    assert any("Keine AprilTags gefunden" in w for w in result.warnings)


def test_tag_seen_in_one_image_is_skipped(room) -> None:
    reconstruction, _ = room
    detections = _observations(reconstruction, {11: 0.150, 12: 0.150})
    only_one = {name: [o for o in obs if o.tag_id == 11] for name, obs in list(detections.items())[:1]}
    only_one = {name: obs for name, obs in only_one.items() if obs}
    assert len(only_one) == 1

    assert triangulate_tags(reconstruction, only_one) == {}


def test_misprinted_tag_is_flagged_as_outlier(room) -> None:
    reconstruction, _ = room
    detections = _observations(reconstruction, {11: 0.150, 12: 0.150, 13: 0.150, 14: 0.120})

    result = tag_scale(triangulate_tags(reconstruction, detections))

    assert result.outliers == (14,)
    assert result.inliers == (11, 12, 13)
    assert result.per_tag[14]["status"] == "outlier"
    assert abs(result.per_tag[14]["deviation_pct"]) > 10.0
    assert result.scale_factor == pytest.approx(TRUE_SCALE, rel=0.005)


def test_edge_lengths_of_a_tilted_square() -> None:
    corners = _tag_corners(11, APRILTAG_SIZE_M, rotation_deg=37.0)

    lengths = tag_edge_lengths(corners)

    assert lengths.shape == (6,)
    assert np.allclose(lengths, expected_edge_lengths(), rtol=1e-9)
    assert np.allclose(expected_edge_lengths()[4:], APRILTAG_SIZE_M * np.sqrt(2.0))


def test_apply_scale_scales_cameras_and_points(room, tmp_path: Path) -> None:
    _, model_dir = room
    out_dir = tmp_path / "scaled" / "model"

    written = apply_scale(model_dir, out_dir, TRUE_SCALE)

    before = pycolmap.Reconstruction(model_dir)
    after = pycolmap.Reconstruction(written)
    assert written == out_dir

    def centre(reconstruction: pycolmap.Reconstruction, image_id: int) -> np.ndarray:
        return np.asarray(reconstruction.image(image_id).cam_from_world().inverse().translation)

    d_before = np.linalg.norm(centre(before, 1) - centre(before, 5))
    d_after = np.linalg.norm(centre(after, 1) - centre(after, 5))
    assert d_after / d_before == pytest.approx(TRUE_SCALE, rel=1e-6)
    ids = sorted(before.points3D)
    assert sorted(after.points3D) == ids
    for point_id in ids[:5]:
        assert np.allclose(np.asarray(after.points3D[point_id].xyz), TRUE_SCALE * np.asarray(before.points3D[point_id].xyz), rtol=1e-6)
    # Orientation is untouched.
    assert np.allclose(after.image(1).cam_from_world().rotation.matrix(), before.image(1).cam_from_world().rotation.matrix(), atol=1e-9)


def test_report_sizes_after_scaling_and_json_export(room, tmp_path: Path) -> None:
    reconstruction, _ = room
    detections = _observations(reconstruction, {11: 0.150, 12: 0.150, 13: 0.150, 14: 0.120})
    geometries = triangulate_tags(reconstruction, detections)
    result = tag_scale(geometries)

    report = build_tag_report(result, geometries, scale_applied=result.scale_factor, mode="scale")

    by_id = {tag["id"]: tag for tag in report["tags"]}
    assert report["mode"] == "scale"
    assert report["scale_factor"] == pytest.approx(TRUE_SCALE, rel=0.005)
    for tag_id in (11, 12, 13):
        assert by_id[tag_id]["size_mm_after"] == pytest.approx(150.0, abs=1.0)
        assert by_id[tag_id]["status"] == "ok"
        assert by_id[tag_id]["images"] >= 2
        corners = np.asarray(by_id[tag_id]["corners_xyz"])
        assert np.linalg.norm(corners[0] - corners[1]) == pytest.approx(0.150, abs=0.003)
        assert np.allclose(by_id[tag_id]["center_xyz"], np.asarray(geometries[tag_id].center_xyz) * result.scale_factor)
    assert by_id[14]["status"] == "outlier"
    assert by_id[14]["size_mm_after"] == pytest.approx(120.0, abs=1.0)

    path = tmp_path / "reports" / "tags.json"
    save_tag_report(path, report)
    import json
    assert json.loads(path.read_text(encoding="utf-8"))["tags"][0]["id"] == 11
