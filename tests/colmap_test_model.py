"""Helpers for writing small, structurally valid COLMAP binary models."""

from __future__ import annotations

import struct
from pathlib import Path


def write_usable_sparse_model(directory: Path, *, point_count: int = 2_000) -> None:
    """Write a one-image COLMAP model that satisfies the sparse quality gate."""
    directory.mkdir(parents=True, exist_ok=True)
    image_points = b"".join(
        struct.pack("<ddQ", float(index), 0.0, index + 1)
        for index in range(point_count)
    )
    (directory / "cameras.bin").write_bytes(
        struct.pack("<QIiQQ4d", 1, 1, 1, 10, 10, 5.0, 5.0, 5.0, 5.0)
    )
    (directory / "images.bin").write_bytes(
        struct.pack("<QIdddddddI", 1, 1, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1)
        + b"image.jpg\0"
        + struct.pack("<Q", point_count)
        + image_points
    )
    point_records = b"".join(
        struct.pack("<QdddBBBdQII", index + 1, float(index), 0.0, 1.0, 255, 255, 255, 0.0, 1, 1, index)
        for index in range(point_count)
    )
    (directory / "points3D.bin").write_bytes(struct.pack("<Q", point_count) + point_records)
