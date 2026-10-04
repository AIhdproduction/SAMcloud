"""Tests for compact point cloud previews and their application endpoint."""

import asyncio
import json
import os
import struct
from pathlib import Path

from samcloud.app.application import pointcloud_preview
from samcloud.app.state import state
from samcloud.viewer.preview import ensure_preview


def _write_ply(path: Path, rows: list[tuple], *, rgb: bool = True, classes: bool = True) -> None:
    properties = ["property float x", "property float y", "property float z"]
    if rgb:
        properties.extend(["property uchar red", "property uchar green", "property uchar blue"])
    if classes:
        properties.append("property uchar scalar_class")
    header = "\n".join([
        "ply", "format binary_little_endian 1.0", f"element vertex {len(rows)}", *properties,
        "end_header", "",
    ]).encode("ascii")
    format_string = "<fff" + ("BBB" if rgb else "") + ("B" if classes else "")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as file:
        file.write(header)
        for row in rows:
            file.write(struct.pack(format_string, *row))


def _read_preview(path: Path) -> tuple[tuple, bytes]:
    content = path.read_bytes()
    return struct.unpack("<4sIII", content[:16]), content[16:]


def test_dense_preview_samples_ply_and_reuses_cache(tmp_path: Path) -> None:
    source = tmp_path / "products" / "classified" / "classified.ply"
    rows = [(float(index), float(index + 1), float(index + 2), index, 20, 30, index % 4) for index in range(7)]
    _write_ply(source, rows)

    preview, count = ensure_preview(tmp_path, "dense", max_points=3)
    header, records = _read_preview(preview)
    assert header == (b"SAMV", 1, 3, 0)
    assert count == 3
    assert len(records) == count * 16
    assert struct.unpack("<fffBBBB", records[:16]) == (0.0, 1.0, 2.0, 0, 20, 30, 0)

    cached_preview, cached_count = ensure_preview(tmp_path, "dense", max_points=3)
    assert cached_preview == preview
    assert cached_count == 3
    assert json.loads((tmp_path / "products" / "viewer" / "dense_preview.json").read_text())["point_count"] == 3

    changed_rows = [(float(index), float(index + 1), float(index + 2), 99, 20, 30, index % 4) for index in range(7)]
    _write_ply(source, changed_rows)
    source_stat = source.stat()
    os.utime(source, ns=(source_stat.st_atime_ns, source_stat.st_mtime_ns + 1))
    _, rebuilt_count = ensure_preview(tmp_path, "dense", max_points=3)
    assert rebuilt_count == 3
    assert _read_preview(preview)[1][12] == 99


def test_dense_preview_defaults_missing_rgb_and_class(tmp_path: Path) -> None:
    source = tmp_path / "products" / "dense" / "fused.ply"
    _write_ply(source, [(1.5, 2.5, 3.5)], rgb=False, classes=False)
    preview, count = ensure_preview(tmp_path, "dense")
    _, records = _read_preview(preview)
    assert count == 1
    assert struct.unpack("<fffBBBB", records) == (1.5, 2.5, 3.5, 200, 200, 200, 0)


def test_sparse_preview_reads_colmap_points(tmp_path: Path) -> None:
    source = tmp_path / "products" / "sparse" / "0" / "points3D.bin"
    source.parent.mkdir(parents=True)
    with source.open("wb") as file:
        file.write(struct.pack("<Q", 3))
        for identifier, x, color, tracks in ((1, 1.0, (10, 20, 30), 1), (2, 2.0, (40, 50, 60), 0), (3, 3.0, (70, 80, 90), 0)):
            file.write(struct.pack("<QdddBBBd", identifier, x, x + 1, x + 2, *color, 0.1))
            file.write(struct.pack("<Q", tracks))
            file.write(b"\x00" * tracks * 8)
    preview, count = ensure_preview(tmp_path, "sparse", max_points=2)
    header, records = _read_preview(preview)
    assert header == (b"SAMV", 1, 2, 0)
    assert count == 2
    assert struct.unpack("<fffBBBB", records[:16]) == (1.0, 2.0, 3.0, 10, 20, 30, 0)
    assert struct.unpack("<fffBBBB", records[16:]) == (3.0, 4.0, 5.0, 70, 80, 90, 0)


def test_pointcloud_endpoint_returns_404_without_project() -> None:
    previous = state.active_project_directory
    state.active_project_directory = None
    try:
        response = asyncio.run(pointcloud_preview("dense"))
    finally:
        state.active_project_directory = previous
    assert response.status_code == 404
    assert json.loads(response.body) == {"error": "No project is open"}
