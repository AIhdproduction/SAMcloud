"""Compact 2D outlines of the screen and ceiling masks of a SAM pass, kept for later 3D fits.

A dark screen leaves almost no dense points, so its instance and fine terms say nothing about its size; its mask in
the photos does.  ``screen_masks.json.gz`` beside the classified cloud holds, per view, the pinhole camera in the
frame of the cloud and the outlines of the recorded masks (a few hundred kB).  A sample of the cloud points tells a
reader whether a product still lies in that frame.
"""

from __future__ import annotations

import gzip
import json
import os
from pathlib import Path

import numpy as np

FILE_NAME = "screen_masks.json.gz"
SCHEMA = 1
INVALID_SUFFIX = ".invalid"
_EDGE_PX = 1.5


def mask_outline(mask, tile_box, threshold: float = 0.5) -> tuple[list[list[float]], bool] | None:
    """(outline in image pixels, whether a tile or the image border cuts it) of the largest part of one mask."""
    import cv2

    x0, y0, x1, y1 = tile_box
    values = cv2.resize(np.asarray(mask, dtype=np.float32), (x1 - x0, y1 - y0), interpolation=cv2.INTER_LINEAR)
    found, _ = cv2.findContours((values >= threshold).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not found:
        return None
    largest = max(found, key=cv2.contourArea)
    ring = cv2.approxPolyDP(largest, _EDGE_PX, True).reshape(-1, 2).astype(np.float64)
    if len(ring) < 3:
        return None
    margin = 3.0
    low, high = ring.min(axis=0), ring.max(axis=0)
    cut = bool(low.min() < margin or high[0] > x1 - x0 - 1 - margin or high[1] > y1 - y0 - 1 - margin)
    return np.round(ring + (x0, y0), 1).tolist(), cut


def view_record(image, camera, width: int, height: int, masks, prompt_texts) -> dict:
    """One view: its camera (frame of the cloud to pixels) and the recorded masks."""
    fx, fy, cx, cy = camera.intrinsics()
    sx, sy = width / float(getattr(camera, "width", width)), height / float(getattr(camera, "height", height))
    origin = np.asarray(image.world_to_cam(np.zeros((1, 3))), dtype=np.float64)[0]
    rotation = (np.asarray(image.world_to_cam(np.eye(3)), dtype=np.float64) - origin).T
    return {
        "name": str(image.name), "size": [int(width), int(height)],
        "pinhole": [fx * sx, fy * sy, cx * sx, cy * sy],
        "world_to_cam": np.column_stack((rotation, origin)).tolist(),
        "masks": [
            {
                "label": str(prompt_texts[item["prompt_idx"]]), "conf": round(float(item["conf"]), 4),
                "cut": bool(item["cut"]), "outline": item["outline"],
            }
            for item in masks
        ],
    }


def write_screen_masks(path: Path, views: list[dict], xyz) -> Path:
    """Write the product beside the classified cloud (atomic)."""
    path = Path(path)
    xyz = np.asarray(xyz)
    sample = xyz[np.linspace(0, len(xyz) - 1, min(len(xyz), 256)).astype(np.int64)] if len(xyz) else xyz
    probe = sample.astype(np.float64)  # only the sample is converted, never the whole cloud
    data = {"schema_version": SCHEMA, "frame_probe": np.round(probe, 4).tolist(), "views": views}
    temporary = path.with_name(path.name + ".tmp")
    with gzip.open(temporary, "wt", encoding="utf-8", compresslevel=6) as handle:
        json.dump(data, handle, separators=(",", ":"))
    os.replace(temporary, path)
    return path


def invalidate_screen_masks(path: Path) -> None:
    """Make an older product unusable after a failure: an empty product replaces it, else a note beside it says so."""
    path = Path(path)
    if not path.is_file():
        return
    try:
        write_screen_masks(path, [], np.empty((0, 3)))
    except Exception:  # noqa: BLE001 - e.g. the product is locked: the note needs no access to it
        path.with_name(path.name + INVALID_SUFFIX).write_text("older than the last SAM pass", encoding="utf-8")


def read_screen_masks(path: Path) -> dict | None:
    """The product, or None when it is absent, unreadable or marked as older than the last SAM pass."""
    try:
        note = Path(path).with_name(Path(path).name + INVALID_SUFFIX)
        if note.is_file() and note.stat().st_mtime_ns >= Path(path).stat().st_mtime_ns:
            return None
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, EOFError, ValueError):
        return None
    return data if isinstance(data, dict) and data.get("schema_version") == SCHEMA else None
