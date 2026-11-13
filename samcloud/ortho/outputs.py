"""Atomic orthophoto artifact writers."""

from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
from typing import Any

import cv2
import numpy as np


def write_outputs(output_dir: Path, *, rgba: np.ndarray, dsm: np.ndarray, dtm: np.ndarray, classes: np.ndarray, validity: np.ndarray, view_count: np.ndarray, metadata: dict[str, Any], fingerprint: str, local_world_file: tuple[float, float, float, float, float, float], original_world_file: tuple[float, float, float, float, float, float] | None, jpeg_sides: tuple[int, ...], jpeg_quality: int, fine_pc: np.ndarray | None = None, fine_pc_confidence: np.ndarray | None = None, fine_labels_source: Path | None = None) -> dict[str, Path]:
    """Write each final artifact through a sibling temporary file then ``os.replace``."""
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    paths["ortho"] = _write_image(output_dir / "ortho.png", cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA), [cv2.IMWRITE_PNG_COMPRESSION, 3])
    paths["dsm"] = _write_npy(output_dir / "dsm.npy", dsm.astype(np.float32))
    paths["dtm"] = _write_npy(output_dir / "dtm.npy", dtm.astype(np.float32))
    paths["classes"] = _write_npy(output_dir / "classes_pc.npy", classes.astype(np.uint8))
    if fine_pc is not None:
        paths["fine_pc"] = _write_npy(output_dir / "fine_pc.npy", np.asarray(fine_pc, dtype=np.uint8))
    if fine_pc_confidence is not None:
        paths["fine_pc_confidence"] = _write_npy(output_dir / "fine_pc_conf.npy", np.asarray(fine_pc_confidence, dtype=np.float16))
    if fine_labels_source is not None and fine_labels_source.is_file():
        paths["fine_pc_labels"] = _write_copy(output_dir / "fine_pc_labels.json", fine_labels_source)
    paths["valid"] = _write_image(output_dir / "valid.png", validity.astype(np.uint8), [cv2.IMWRITE_PNG_COMPRESSION, 3])
    paths["view_count"] = _write_image(output_dir / "view_count.png", view_count.astype(np.uint8), [cv2.IMWRITE_PNG_COMPRESSION, 3])
    paths["local_world"] = _write_text(output_dir / "ortho_local.pgw", _world_text(local_world_file))
    if original_world_file is not None:
        paths["world"] = _write_text(output_dir / "ortho.pgw", _world_text(original_world_file))
    rgb = _white_background(rgba)
    for side in jpeg_sides:
        image = _resize_long_side(rgb, side)
        key = f"jpeg_{side}"
        paths[key] = _write_image(output_dir / f"ortho_{side}.jpg", cv2.cvtColor(image, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
    preview_key = f"jpeg_{jpeg_sides[-1]}"
    paths["preview"] = _write_copy(output_dir / "preview.jpg", paths[preview_key])
    metadata_path = output_dir / "ortho.json"
    paths["metadata"] = _write_text(metadata_path, json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    paths["fingerprint"] = _write_text(output_dir / "fingerprint.txt", fingerprint + "\n")
    return paths


def _white_background(rgba: np.ndarray) -> np.ndarray:
    alpha = rgba[..., 3:4].astype(np.float32) / 255.0
    return np.round(rgba[..., :3].astype(np.float32) * alpha + 255.0 * (1.0 - alpha)).astype(np.uint8)


def _resize_long_side(image: np.ndarray, side: int) -> np.ndarray:
    height, width = image.shape[:2]
    factor = min(1.0, side / max(width, height))
    if factor == 1.0:
        return image
    return cv2.resize(image, (max(1, round(width * factor)), max(1, round(height * factor))), interpolation=cv2.INTER_AREA)


def _world_text(values: tuple[float, float, float, float, float, float]) -> str:
    return "".join(f"{value:.15g}\n" for value in values)


def _temporary(target: Path) -> Path:
    descriptor, name = tempfile.mkstemp(prefix=f".{target.stem}.", suffix=f".tmp{target.suffix}", dir=target.parent)
    os.close(descriptor)
    return Path(name)


def _write_image(target: Path, image: np.ndarray, params: list[int]) -> Path:
    temporary = _temporary(target)
    if not cv2.imwrite(str(temporary), image, params):
        raise OSError(f"Could not write image: {target}")
    os.replace(temporary, target)
    return target


def _write_npy(target: Path, value: np.ndarray) -> Path:
    temporary = _temporary(target)
    with temporary.open("wb") as handle:
        np.save(handle, value, allow_pickle=False)
    os.replace(temporary, target)
    return target


def _write_text(target: Path, value: str) -> Path:
    temporary = _temporary(target)
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, target)
    return target


def _write_copy(target: Path, source: Path) -> Path:
    temporary = _temporary(target)
    temporary.write_bytes(source.read_bytes())
    os.replace(temporary, target)
    return target
