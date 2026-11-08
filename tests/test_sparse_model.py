from __future__ import annotations

import json
import struct
from pathlib import Path

from samcloud.core.sparse_model import resolve_sparse_model, sparse_points_path
from samcloud.viewer.preview import ensure_preview, _source_path


def test_sparse_model_prefers_gcp_adjusted_before_final_model(tmp_path: Path) -> None:
    sparse = tmp_path / "products" / "sparse"
    _write_points(sparse / "gcp_adjusted" / "points3D.bin", 1.0)
    _write_points(sparse / "0_ba_1_ba_2" / "points3D.bin", 2.0)
    _write_points(sparse / "0_geo" / "points3D.bin", 3.0)
    (sparse / "final_model.txt").write_text("0_ba_1_ba_2\n", encoding="utf-8")

    assert resolve_sparse_model(sparse) == sparse / "gcp_adjusted"


def test_sparse_model_uses_configured_final_before_legacy_and_fragments(tmp_path: Path) -> None:
    sparse = tmp_path / "products" / "sparse"
    _write_points(sparse / "0_ba_1_ba_2" / "points3D.bin", 2.0)
    _write_points(sparse / "0_geo" / "points3D.bin", 3.0)
    _write_points(sparse / "0" / "points3D.bin", 4.0)
    _write_points(sparse / "1" / "points3D.bin", 5.0)
    (sparse / "final_model.txt").write_text("0_ba_1_ba_2\n", encoding="utf-8")

    assert sparse_points_path(sparse) == sparse / "0_ba_1_ba_2" / "points3D.bin"


def test_sparse_model_ignores_invalid_configured_target_and_uses_remaining_model(tmp_path: Path) -> None:
    sparse = tmp_path / "products" / "sparse"
    (sparse / "configured_empty").mkdir(parents=True)
    _write_points(sparse / "2" / "points3D.bin", 5.0)
    (sparse / "final_model.txt").write_text("configured_empty\n", encoding="utf-8")

    assert resolve_sparse_model(sparse) == sparse / "2"


def test_sparse_preview_signature_switches_to_configured_final_model(tmp_path: Path) -> None:
    project = tmp_path / "project"
    sparse = project / "products" / "sparse"
    _write_points(sparse / "0_ba_1" / "points3D.bin", 1.0)
    _write_points(sparse / "0_ba_2" / "points3D.bin", 9.0)
    marker = sparse / "final_model.txt"
    marker.write_text("0_ba_1\n", encoding="utf-8")

    preview, _count = ensure_preview(project, "sparse")
    first_signature = json.loads(preview.with_suffix(".json").read_text(encoding="utf-8"))["signature"]
    assert struct.unpack("<f", preview.read_bytes()[16:20])[0] == 1.0

    marker.write_text("0_ba_2\n", encoding="utf-8")
    refreshed, _count = ensure_preview(project, "sparse")
    second_signature = json.loads(refreshed.with_suffix(".json").read_text(encoding="utf-8"))["signature"]
    assert _source_path(project, "sparse") == sparse / "0_ba_2" / "points3D.bin"
    assert first_signature["source"] != second_signature["source"]
    assert struct.unpack("<f", refreshed.read_bytes()[16:20])[0] == 9.0


def _write_points(path: Path, x: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(struct.pack("<Q", 1) + struct.pack("<QdddBBBdQ", 1, x, 0.0, 0.0, 1, 2, 3, 0.0, 0))
