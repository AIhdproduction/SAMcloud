"""Compact, cacheable point cloud previews for the browser viewer."""

from __future__ import annotations

import json
import math
import os
import struct
from pathlib import Path

import numpy as np


_HEADER = struct.Struct("<4sIII")
_POINT = np.dtype(
    [("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("red", "u1"),
     ("green", "u1"), ("blue", "u1"), ("class_id", "u1")]
)
_PLY_TYPES = {
    "char": "i1", "int8": "i1", "uchar": "u1", "uint8": "u1",
    "short": "<i2", "int16": "<i2", "ushort": "<u2", "uint16": "<u2",
    "int": "<i4", "int32": "<i4", "uint": "<u4", "uint32": "<u4",
    "float": "<f4", "float32": "<f4", "double": "<f8", "float64": "<f8",
}


def ensure_preview(
    project_dir: Path | str, kind: str, max_points: int = 300_000
) -> tuple[Path, int]:
    """Return a validated browser preview, rebuilding it when its source changed."""
    if kind not in {"dense", "sparse"}:
        raise ValueError("Point cloud kind must be 'dense' or 'sparse'")
    if max_points < 1:
        raise ValueError("max_points must be at least 1")

    project_dir = Path(project_dir)
    source = _find_source(project_dir, kind)
    source_stat = source.stat()
    signature = {
        "source": str(source.relative_to(project_dir)).replace("\\", "/"),
        "size": source_stat.st_size,
        "mtime_ns": source_stat.st_mtime_ns,
        "max_points": max_points,
    }
    viewer_dir = project_dir / "products" / "viewer"
    preview = viewer_dir / f"{kind}_preview.bin"
    metadata = viewer_dir / f"{kind}_preview.json"
    cached_count = _cached_count(preview, metadata, signature)
    if cached_count is not None:
        return preview, cached_count

    viewer_dir.mkdir(parents=True, exist_ok=True)
    temporary = preview.with_suffix(".bin.tmp")
    try:
        if kind == "dense":
            point_count = _write_ply_preview(source, temporary, max_points)
        else:
            point_count = _write_sparse_preview(source, temporary, max_points)
        os.replace(temporary, preview)
        metadata_temporary = metadata.with_suffix(".json.tmp")
        metadata_temporary.write_text(
            json.dumps({"signature": signature, "point_count": point_count}), encoding="utf-8"
        )
        os.replace(metadata_temporary, metadata)
    finally:
        temporary.unlink(missing_ok=True)
    return preview, point_count


def _find_source(project_dir: Path, kind: str) -> Path:
    candidates = (
        (project_dir / "products" / "classified" / "classified.ply",
         project_dir / "products" / "dense" / "fused.ply")
        if kind == "dense"
        else (project_dir / "products" / "sparse" / "0_geo" / "points3D.bin",
              project_dir / "products" / "sparse" / "0" / "points3D.bin")
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"No {kind} point cloud source was found")


def _cached_count(preview: Path, metadata: Path, signature: dict[str, object]) -> int | None:
    try:
        data = json.loads(metadata.read_text(encoding="utf-8"))
        count = int(data["point_count"])
        if data.get("signature") != signature or count < 0 or not preview.is_file():
            return None
        with preview.open("rb") as file:
            magic, version, stored_count, _ = _HEADER.unpack(file.read(_HEADER.size))
        if magic != b"SAMV" or version != 1 or stored_count != count:
            return None
        if preview.stat().st_size != _HEADER.size + count * _POINT.itemsize:
            return None
        return count
    except (OSError, ValueError, KeyError, json.JSONDecodeError, struct.error):
        return None


def _read_ply_header(file) -> tuple[int, np.dtype]:
    first_line = file.readline().decode("ascii", errors="replace").strip()
    if first_line != "ply":
        raise ValueError("PLY source has no valid header")
    format_line = file.readline().decode("ascii", errors="replace").strip().split()
    if format_line[:2] != ["format", "binary_little_endian"]:
        raise ValueError("Only binary little-endian PLY files are supported")

    vertex_count: int | None = None
    vertex_properties: list[tuple[str, str]] = []
    current_element: str | None = None
    while True:
        raw_line = file.readline()
        if not raw_line:
            raise ValueError("PLY header ends unexpectedly")
        parts = raw_line.decode("ascii", errors="replace").strip().split()
        if not parts or parts[0] in {"comment", "obj_info"}:
            continue
        if parts[0] == "end_header":
            break
        if parts[:1] == ["element"] and len(parts) == 3:
            current_element = parts[1]
            if current_element == "vertex":
                vertex_count = int(parts[2])
            continue
        if parts[:1] == ["property"] and current_element == "vertex":
            if len(parts) == 3:
                type_name, field_name = parts[1:]
                if type_name not in _PLY_TYPES:
                    raise ValueError(f"Unsupported PLY property type '{type_name}'")
                vertex_properties.append((field_name, _PLY_TYPES[type_name]))
            elif len(parts) >= 5 and parts[1] == "list":
                raise ValueError("PLY vertex list properties are not supported")
    if vertex_count is None or vertex_count < 0 or not vertex_properties:
        raise ValueError("PLY header has no usable vertex element")
    return vertex_count, np.dtype(vertex_properties)


def _write_ply_preview(source: Path, target: Path, max_points: int) -> int:
    with source.open("rb") as input_file:
        total, source_dtype = _read_ply_header(input_file)
        for name in ("x", "y", "z"):
            if name not in source_dtype.names:
                raise ValueError(f"PLY source is missing coordinate '{name}'")
        step = max(1, math.ceil(total / max_points))
        count = (total + step - 1) // step
        with target.open("wb") as output_file:
            output_file.write(_HEADER.pack(b"SAMV", 1, count, 0))
            for start in range(0, total, 250_000):
                rows = np.fromfile(input_file, dtype=source_dtype, count=min(250_000, total - start))
                rows = rows[(np.arange(len(rows)) + start) % step == 0]
                _write_records(output_file, rows)
    return count


def _write_records(output_file, rows: np.ndarray) -> None:
    records = np.empty(len(rows), dtype=_POINT)
    records["x"] = rows["x"]
    records["y"] = rows["y"]
    records["z"] = rows["z"]
    for output_name, source_name in (("red", "red"), ("green", "green"), ("blue", "blue")):
        records[output_name] = np.clip(rows[source_name], 0, 255) if source_name in rows.dtype.names else 200
    records["class_id"] = (
        np.clip(rows["scalar_class"], 0, 255) if "scalar_class" in rows.dtype.names else 0
    )
    records.tofile(output_file)


def _write_sparse_preview(source: Path, target: Path, max_points: int) -> int:
    point_header = struct.Struct("<QdddBBBd")
    with source.open("rb") as input_file:
        raw_count = input_file.read(8)
        if len(raw_count) != 8:
            raise ValueError("COLMAP points3D file is truncated")
        total = struct.unpack("<Q", raw_count)[0]
        step = max(1, math.ceil(total / max_points))
        count = (total + step - 1) // step
        with target.open("wb") as output_file:
            output_file.write(_HEADER.pack(b"SAMV", 1, count, 0))
            records = np.empty(min(count, 100_000), dtype=_POINT)
            buffered = 0
            for index in range(total):
                entry = input_file.read(point_header.size)
                if len(entry) != point_header.size:
                    raise ValueError("COLMAP points3D file is truncated")
                _, x, y, z, red, green, blue, _ = point_header.unpack(entry)
                raw_track_length = input_file.read(8)
                if len(raw_track_length) != 8:
                    raise ValueError("COLMAP points3D file is truncated")
                track_length = struct.unpack("<Q", raw_track_length)[0]
                input_file.seek(track_length * 8, os.SEEK_CUR)
                if index % step != 0:
                    continue
                records[buffered] = (x, y, z, red, green, blue, 0)
                buffered += 1
                if buffered == len(records):
                    records.tofile(output_file)
                    buffered = 0
            records[:buffered].tofile(output_file)
    return count
