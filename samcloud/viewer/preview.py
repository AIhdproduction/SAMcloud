"""Create compact browser previews from COLMAP point cloud products."""

from __future__ import annotations

import json
import math
import struct
from pathlib import Path

import numpy as np


PREVIEW_MAGIC = b"SAMV"
PREVIEW_VERSION = 1
PREVIEW_RECORD = np.dtype(
    [
        ("x", "<f4"),
        ("y", "<f4"),
        ("z", "<f4"),
        ("red", "u1"),
        ("green", "u1"),
        ("blue", "u1"),
        ("class_id", "u1"),
    ],
    align=False,
)


def ensure_preview(project_directory: Path, kind: str, *, max_points: int = 300_000) -> tuple[Path, int]:
    """Return a cached compact point preview for ``kind`` (``sparse`` or ``dense``)."""
    if kind not in {"sparse", "dense"}:
        raise ValueError("Unsupported point cloud kind")
    source = _source_path(project_directory, kind)
    if source is None:
        raise FileNotFoundError(f"No {kind} point cloud is available yet")
    output = project_directory / "products" / "viewer" / f"{kind}_preview.bin"
    metadata_path = output.with_suffix(".json")
    signature = {
        "source": str(source),
        "size": source.stat().st_size,
        "mtime_ns": source.stat().st_mtime_ns,
        "max_points": max_points,
    }
    if output.exists() and metadata_path.exists():
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if metadata.get("signature") == signature:
                return output, int(metadata["point_count"])
        except (OSError, ValueError, TypeError):
            pass

    output.parent.mkdir(parents=True, exist_ok=True)
    if kind == "sparse":
        point_count = _write_sparse_preview(source, output, max_points)
    else:
        point_count = _write_ply_preview(source, output, max_points)
    metadata_path.write_text(
        json.dumps({"signature": signature, "point_count": point_count}, indent=2),
        encoding="utf-8",
    )
    return output, point_count


def _source_path(project_directory: Path, kind: str) -> Path | None:
    products = project_directory / "products"
    if kind == "sparse":
        candidates = (
            products / "sparse" / "gcp_adjusted" / "points3D.bin",
            products / "sparse" / "0_geo" / "points3D.bin",
            products / "sparse" / "0" / "points3D.bin",
        )
    else:
        candidates = (
            products / "classified" / "classified.ply",
            products / "dense" / "fused.ply",
        )
    return next((path for path in candidates if path.exists() and path.stat().st_size > 0), None)


def _write_header(file, point_count: int) -> None:
    file.write(PREVIEW_MAGIC)
    # Keep the payload aligned to a 16-byte record boundary.
    file.write(struct.pack("<III", PREVIEW_VERSION, point_count, 0))


def _write_sparse_preview(source: Path, output: Path, max_points: int) -> int:
    with source.open("rb") as source_file:
        total = struct.unpack("<Q", source_file.read(8))[0]
        step = max(1, math.ceil(total / max_points))
        selected = math.ceil(total / step)
        with output.open("wb") as output_file:
            _write_header(output_file, selected)
            for index in range(total):
                point_id, x, y, z, red, green, blue, _error = struct.unpack(
                    "<QdddBBBd", source_file.read(43)
                )
                track_length = struct.unpack("<Q", source_file.read(8))[0]
                source_file.seek(track_length * 8, 1)
                if index % step == 0:
                    output_file.write(
                        struct.pack("<fffBBBB", x, y, z, red, green, blue, 0)
                    )
    return selected


def _read_ply_header(source: Path) -> tuple[int, int, np.dtype, dict[str, int]]:
    with source.open("rb") as file:
        header = bytearray()
        while b"end_header" not in header:
            line = file.readline()
            if not line:
                raise ValueError("PLY header is incomplete")
            header.extend(line)
        header_text = header.decode("ascii")
    lines = [line.strip().split() for line in header_text.splitlines()]
    if len(lines) < 3 or lines[0] != ["ply"] or lines[1][:2] != ["format", "binary_little_endian"]:
        raise ValueError("Only binary little-endian PLY files are supported")
    vertex_count = 0
    properties: list[tuple[str, str]] = []
    in_vertex = False
    for parts in lines:
        if parts[:2] == ["element", "vertex"]:
            vertex_count = int(parts[2])
            in_vertex = True
        elif parts[:2] == ["element"]:
            in_vertex = False
        elif in_vertex and parts[:1] == ["property"] and len(parts) == 3:
            properties.append((parts[1], parts[2]))
    type_map = {"float": "<f4", "float32": "<f4", "uchar": "u1", "uint8": "u1"}
    if not vertex_count or not properties or any(kind not in type_map for kind, _ in properties):
        raise ValueError("PLY does not contain a supported vertex element")
    dtype = np.dtype([(name, type_map[kind]) for kind, name in properties], align=False)
    offsets = {name: index for index, (_, name) in enumerate(properties)}
    return vertex_count, len(properties), dtype, offsets


def _write_ply_preview(source: Path, output: Path, max_points: int) -> int:
    total, _property_count, dtype, offsets = _read_ply_header(source)
    required = {"x", "y", "z"}
    if not required.issubset(offsets):
        raise ValueError("PLY is missing point coordinates")
    step = max(1, math.ceil(total / max_points))
    selected = math.ceil(total / step)
    header_size = _ply_header_size(source)
    with source.open("rb") as source_file, output.open("wb") as output_file:
        source_file.seek(header_size)
        _write_header(output_file, selected)
        written = 0
        while written < total:
            chunk = np.fromfile(source_file, dtype=dtype, count=min(250_000, total - written))
            if not len(chunk):
                break
            indices = np.flatnonzero((np.arange(len(chunk), dtype=np.int64) + written) % step == 0)
            for point in chunk[indices]:
                red = int(point["red"]) if "red" in offsets else 200
                green = int(point["green"]) if "green" in offsets else 200
                blue = int(point["blue"]) if "blue" in offsets else 200
                class_id = int(point["scalar_class"]) if "scalar_class" in offsets else 0
                output_file.write(
                    struct.pack(
                        "<fffBBBB",
                        float(point["x"]), float(point["y"]), float(point["z"]),
                        red, green, blue, class_id,
                    )
                )
            written += len(chunk)
    return selected


def _ply_header_size(source: Path) -> int:
    with source.open("rb") as file:
        size = 0
        while True:
            line = file.readline()
            if not line:
                raise ValueError("PLY header is incomplete")
            size += len(line)
            if line.strip() == b"end_header":
                return size
