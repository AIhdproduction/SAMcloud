"""AprilTag additions to the control-point review payload and image resolution."""

from __future__ import annotations

import json
from pathlib import Path

from samcloud.app.gcp_review import marker_payload, resolve_project_image


def _detection() -> dict:
    return {
        "role": "pass", "enabled": True, "status": "accepted", "triangulated_model_xyz": [1.0, 2.0, 3.0],
        "observations": [{"image": "front/abc.jpg", "u": 10.0, "v": 20.0, "source": "auto"}],
    }


def test_marker_payload_adds_apriltag_fields_only_for_tags(tmp_path: Path) -> None:
    (tmp_path / "control_points").mkdir()
    (tmp_path / "reports").mkdir()
    (tmp_path / "control_points" / "detections.json").write_text(
        json.dumps({"PP1": _detection(), "TAG3": _detection()}), encoding="utf-8",
    )
    (tmp_path / "reports" / "apriltag_report.json").write_text(json.dumps({
        "mode": "scale", "scale_factor": 2.0, "warnings": [],
        "tags": [{"id": 3, "images": 4, "size_mm_before": 70.0, "size_mm_after": 149.5, "deviation_pct": -0.33, "status": "ok"}],
    }), encoding="utf-8")

    markers = {marker["name"]: marker for marker in marker_payload(tmp_path)["markers"]}

    assert markers["TAG3"]["kind"] == "apriltag"
    assert markers["TAG3"]["tag_id"] == 3
    assert markers["TAG3"]["tag_size_mm"] == 149.5
    assert markers["TAG3"]["tag_deviation_pct"] == -0.33
    for key in ("kind", "tag_id", "tag_size_mm", "tag_deviation_pct"):
        assert key not in markers["PP1"]


def test_resolve_project_image_finds_cubemap_face(tmp_path: Path) -> None:
    face = tmp_path / "inputs" / "cubemap" / "front" / "abc.jpg"
    face.parent.mkdir(parents=True)
    face.write_bytes(b"jpg")

    assert resolve_project_image(tmp_path, "front/abc.jpg", "original") == face.resolve()
