"""Behaviour tests for dense-cloud instance metadata and point overlays."""

from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np
import pytest

from samcloud.app import application
from samcloud.app.state import state


_DTYPE = np.dtype(
    [
        ("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
        ("red", "u1"), ("green", "u1"), ("blue", "u1"), ("scalar_class", "<i4"), ("instance_id", "<u4"),
    ],
    align=False,
)


def _write_classified(project: Path, points: list[tuple[float, float, float, int, int]]) -> Path:
    path = project / "products" / "classified" / "classified.ply"
    path.parent.mkdir(parents=True, exist_ok=True)
    data = np.zeros(len(points), dtype=_DTYPE)
    for index, (x, y, z, red, instance_id) in enumerate(points):
        data[index]["x"], data[index]["y"], data[index]["z"] = x, y, z
        data[index]["red"], data[index]["green"], data[index]["blue"] = red, red + 1, red + 2
        data[index]["scalar_class"], data[index]["instance_id"] = 4, instance_id
    header = (
        f"ply\nformat binary_little_endian 1.0\nelement vertex {len(data)}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        "property int scalar_class\nproperty uint instance_id\nend_header\n"
    ).encode("ascii")
    path.write_bytes(header + data.tobytes())
    return path


def _read_points(response) -> list[tuple[float, float, float, int, int, int]]:
    content = response.body
    assert content[:4] == b"SAMV"
    count = struct.unpack_from("<I", content, 8)[0]
    assert len(content) == 16 + count * 15
    return [struct.unpack_from("<fffBBB", content, 16 + index * 15) for index in range(count)]


@pytest.fixture
def active_project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    project = tmp_path / "project"
    monkeypatch.setattr(state, "active_project_directory", project)
    return project


def test_instances_api_lists_metadata_and_is_empty_without_manifest(active_project: Path) -> None:
    assert json.loads(application.instances().body) == {"instances": []}
    manifest = {
        "schema_version": 1,
        "instances": [{"id": 7, "class_id": 17, "class_name": "furniture", "label": "chair", "label_index": 3,
                       "point_count": 2, "bbox_min": [1, 2, 3], "bbox_max": [4, 5, 6]}],
    }
    path = active_project / "products" / "classified" / "instances.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(manifest), encoding="utf-8")

    payload = json.loads(application.instances().body)
    assert payload["instances"] == [{"id": 7, "label": "chair", "label_de": "Stuhl", "label_index": 3,
                                      "class_id": 17, "class_name": "furniture", "point_count": 2,
                                      "bbox_min": [1, 2, 3], "bbox_max": [4, 5, 6]}]


def test_instance_points_api_returns_only_requested_points_in_ply_frame(active_project: Path) -> None:
    _write_classified(active_project, [(10.12345, -3.67891, 5.25, 10, 7), (99.0, 99.0, 99.0, 40, 0), (2.5, 8.75, -4.5, 30, 7)])

    response = application.instance_points(7)
    points = _read_points(response)

    assert response.status_code == 200
    assert len(points) == 2
    assert np.allclose(np.asarray([point[:3] for point in points]), [[10.12345, -3.67891, 5.25], [2.5, 8.75, -4.5]], atol=1e-4)
    assert [point[3:] for point in points] == [(10, 11, 12), (30, 31, 32)]
    assert application.instance_points(99).status_code == 404


def test_instance_points_api_caps_uniformly_and_rebuilds_stale_index(active_project: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(application, "_INSTANCE_POINT_LIMIT", 3)
    source = _write_classified(active_project, [(float(index), 0.0, 0.0, index, 4) for index in range(8)])
    capped = _read_points(application.instance_points(4))
    assert len(capped) == 3
    assert [point[0] for point in capped] == [0.0, 3.0, 7.0]

    cache = active_project / "products" / "viewer" / "instance_index.npz"
    with np.load(cache) as before:
        before_signature = before["signature"].copy()
    _write_classified(active_project, [(1.0, 2.0, 3.0, 21, 9)])
    assert source.is_file()
    rebuilt = _read_points(application.instance_points(9))
    with np.load(cache) as after:
        after_signature = after["signature"].copy()
    assert len(rebuilt) == 1
    assert np.allclose(rebuilt[0][:3], [1.0, 2.0, 3.0], atol=1e-4)
    assert not np.array_equal(before_signature, after_signature)


def test_instance_points_api_binary_layout_and_default_colours(active_project: Path) -> None:
    _write_classified(active_project, [(1.5, 2.5, 3.5, 10, 2), (4.0, 5.0, 6.0, 20, 2)])
    body = application.instance_points(2).body
    assert body[:4] == b"SAMV"
    version, count, reserved = np.frombuffer(body[4:16], dtype="<u4")
    assert (version, count, reserved) == (1, 2, 0)
    assert len(body) == 16 + 15 * 2


def test_instances_api_label_de_falls_back_to_german_class_name(active_project: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = {"instance_labels_de": {"chair": "Stuhl"}, "labels_de": {"indoor": ["Unbekannt", "Wand", "Fenster"]}}
    monkeypatch.setattr("samcloud.classification.custom.load_project_classes", lambda _directory: config)
    monkeypatch.setattr(state, "active_project", lambda: {"settings": {"classification": {"class_set": "indoor"}}})
    manifest = {"instances": [
        {"id": 1, "class_id": 2, "class_name": "window", "label": "casement window", "label_index": 2},
        {"id": 2, "class_id": 2, "class_name": "window", "label": "chair", "label_index": 1},
        {"id": 3, "class_id": 99, "class_name": "x", "label": "mystery", "label_index": 1},
    ]}
    path = active_project / "products" / "classified" / "instances.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(manifest), encoding="utf-8")

    labels = [item["label_de"] for item in json.loads(application.instances().body)["instances"]]
    assert labels == ["Fenster", "Stuhl", "mystery"]

    monkeypatch.setattr("samcloud.classification.custom.load_project_classes", lambda _directory: {})
    labels = [item["label_de"] for item in json.loads(application.instances().body)["instances"]]
    assert labels == ["casement window", "chair", "mystery"]
