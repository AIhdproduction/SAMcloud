"""Tests for sparse camera metadata, rig grouping, and safe thumbnails."""

from __future__ import annotations

import json
from pathlib import Path
import struct

import cv2
import numpy as np
import pycolmap
import pytest

from samcloud.app import application
from samcloud.app.state import state
from samcloud.viewer.cameras import camera_payload, camera_thumbnail
from samcloud.viewer.preview import ensure_preview


def _rig_project(tmp_path: Path) -> Path:
    project = tmp_path / "camera-project"
    model = project / "products" / "sparse" / "0"
    model.mkdir(parents=True)
    reconstruction = pycolmap.Reconstruction()
    for camera_id in (1, 2):
        reconstruction.add_camera(pycolmap.Camera(
            camera_id=camera_id, model="SIMPLE_PINHOLE", width=20, height=10, params=[10.0, 10.0, 5.0],
        ))
    sensor_one = pycolmap.sensor_t(pycolmap.SensorType.CAMERA, 1)
    sensor_two = pycolmap.sensor_t(pycolmap.SensorType.CAMERA, 2)
    rig = pycolmap.Rig(rig_id=7)
    rig.add_ref_sensor(sensor_one)
    rig.add_sensor(sensor_two, pycolmap.Rigid3d(translation=[1.0, 0.0, 0.0]))
    reconstruction.add_rig(rig)
    for frame_id, centre in enumerate(((10.0, 20.0, 30.0), (11.0, 20.0, 30.0)), start=1):
        frame = pycolmap.Frame(
            frame_id=frame_id,
            rig_id=7,
            rig_from_world=pycolmap.Rigid3d(translation=-np.asarray(centre)),
        )
        first_image_id, second_image_id = frame_id * 10 + 1, frame_id * 10 + 2
        frame.add_data_id(pycolmap.data_t(sensor_one, first_image_id))
        frame.add_data_id(pycolmap.data_t(sensor_two, second_image_id))
        frame.finalize_data_ids()
        reconstruction.add_frame(frame)
        suffix = f"panorama_{frame_id - 1:06d}.jpg"
        reconstruction.add_image(pycolmap.Image(
            name=f"front/{suffix}", camera_id=1, image_id=first_image_id, frame_id=frame_id,
        ))
        reconstruction.add_image(pycolmap.Image(
            name=f"right/{suffix}", camera_id=2, image_id=second_image_id, frame_id=frame_id,
        ))
        reconstruction.register_frame(frame_id)
    reconstruction.add_point3D(np.array([3.0, 4.0, 5.0]), pycolmap.Track())
    reconstruction.write(model)

    frames = project / "inputs" / "panorama_frames"
    frames.mkdir(parents=True)
    manifest = []
    for index in range(2):
        source = frames / f"frame_{index:06d}.jpg"
        assert cv2.imwrite(str(source), np.full((20, 40, 3), index * 40 + 30, dtype=np.uint8))
        manifest.append({"source_id": f"panorama_{index:06d}", "source_path": str(source), "source_kind": "video_frame"})
    (project / "inputs" / "panorama_manifest.json").write_text(json.dumps({"panoramas": manifest}), encoding="utf-8")
    (project / "project.samcloud.yml").write_text("acquisition:\n  type: panorama_360\n", encoding="utf-8")
    return project


def _response_json(response) -> dict[str, object]:
    return json.loads(response.body)


def test_sparse_viewer_exposes_camera_controls() -> None:
    from samcloud.app import pointcloud_viewer, potree_viewer

    config = potree_viewer._viewer_config("sparse", "Sparse", [], camera_endpoint="/api/cameras/sparse")
    html = potree_viewer._html("viewer", show_classes=False, show_cameras=True)
    script = potree_viewer._script(config)
    fallback_script = pointcloud_viewer._script(
        "viewer", "/api/pointcloud/sparse", camera_endpoint="/api/cameras/sparse",
    )

    assert config["camera_endpoint"] == "/api/cameras/sparse"
    assert 'data-action="cameras"' in html
    assert 'data-action="camera-size"' in html
    assert 'min="0.25" max="0.75" step="0.25"' in html
    assert 'data-action="camera-sphere-texture"' in html
    for removed in ("camera-path", "camera-photos", "camera-path-width", "camera-photo-size"):
        assert f'data-action="{removed}"' not in html
    assert "addEventListener(\"dblclick\"" in script
    assert "SphereGeometry" in script
    assert "Foto anzeigen" in script
    assert "/api/camera-thumbnail/" in script
    assert "samcloud:sparse-camera-settings:v1" in script
    assert "samcloud:sparse-camera-settings:v1" in fallback_script
    assert "slice(0, 72)" in fallback_script


def test_non_rig_model_returns_one_camera_per_registered_image_sorted_by_name(tmp_path: Path) -> None:
    project = tmp_path / "normal-project"
    model = project / "products" / "sparse" / "0"
    model.mkdir(parents=True)
    (project / "inputs" / "images").mkdir(parents=True)
    assert cv2.imwrite(str(project / "inputs" / "images" / "camera.jpg"), np.full((10, 10, 3), 80, dtype=np.uint8))
    (model / "cameras.bin").write_bytes(struct.pack("<QIiQQ4d", 1, 1, 1, 10, 10, 5.0, 5.0, 5.0, 5.0))
    (model / "images.bin").write_bytes(
        struct.pack("<QIdddddddI", 1, 1, 1.0, 0.0, 0.0, 0.0, -4.0, -5.0, -6.0, 1)
        + b"camera.jpg\0" + struct.pack("<Q", 0)
    )
    (model / "points3D.bin").write_bytes(struct.pack("<Q", 0))

    payload = camera_payload(project)

    assert payload["path_available"] is False
    assert payload["cameras"] == [
        {
            "id": "image-1", "image_name": "camera.jpg", "image_path": str((project / "inputs" / "images" / "camera.jpg").resolve()),
            "source_name": "camera.jpg", "position": [4.0, 5.0, 6.0], "direction": [0.0, 0.0, 1.0],
            "frame_id": None, "index": 0, "points3d": 0, "mean_reprojection_error": None,
        }
    ]


def test_rig_model_returns_one_camera_per_frame_in_sparse_preview_frame(tmp_path: Path) -> None:
    project = _rig_project(tmp_path)

    payload = camera_payload(project)

    assert payload["path_available"] is True
    cameras = payload["cameras"]
    assert len(cameras) == 2
    assert [camera["id"] for camera in cameras] == ["frame-1", "frame-2"]
    np.testing.assert_allclose([camera["position"] for camera in cameras], [[10.0, 20.0, 30.0], [11.0, 20.0, 30.0]])
    assert [camera["source_name"] for camera in cameras] == ["frame_000000.jpg", "frame_000001.jpg"]
    assert all(len(camera["direction"]) == 3 for camera in cameras)

    preview, point_count = ensure_preview(project, "sparse")
    assert point_count == 1
    raw = preview.read_bytes()
    assert struct.unpack_from("<fff", raw, 16) == pytest.approx((3.0, 4.0, 5.0))


def test_camera_thumbnail_is_id_addressed_and_cached(tmp_path: Path) -> None:
    project = _rig_project(tmp_path)
    camera_id = camera_payload(project)["cameras"][0]["id"]

    first = camera_thumbnail(project, camera_id, 320)
    second = camera_thumbnail(project, camera_id, 320)

    assert first.startswith(b"\xff\xd8")
    assert second == first
    with pytest.raises(FileNotFoundError):
        camera_thumbnail(project, "../frame-1", 320)


def test_camera_endpoints_return_payload_and_reject_unknown_thumbnail(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    project = _rig_project(tmp_path)
    monkeypatch.setattr(state, "active_project_directory", project)

    response = application.sparse_cameras()
    assert response.status_code == 200
    assert len(_response_json(response)["cameras"]) == 2

    thumbnail = application.sparse_camera_thumbnail("frame-1", "320")
    assert thumbnail.status_code == 200
    assert thumbnail.media_type == "image/jpeg"
    assert application.sparse_camera_thumbnail("..", "320").status_code == 404
