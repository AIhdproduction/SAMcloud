"""Create the class scheme consumed by the Potree viewer integration."""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
from typing import Any

import laspy
import numpy as np

from samcloud.core.paths import PROJECT_ROOT
from samcloud.viewer.preview import _ply_header_size, _read_ply_header, _source_path

POTREE_CONVERTER = PROJECT_ROOT / "tools" / "PotreeConverter" / "PotreeConverter.exe"
POTREE_LIBRARY = PROJECT_ROOT / "assets" / "potree" / "build" / "potree" / "potree.js"
_NO_CONSOLE = 0x08000000 if os.name == "nt" else 0


def write_viewer_manifest(path: Path, classes: list[dict[str, Any]], point_cloud_url: str) -> None:
    """Write a viewer manifest containing labels, colors, and class visibility."""
    manifest = {
        "point_cloud_url": point_cloud_url,
        "display_mode": "rgb",
        "classes": classes,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def potree_available() -> bool:
    """Potree needs both the browser library and the converter."""
    return POTREE_LIBRARY.is_file() and POTREE_CONVERTER.is_file()


def write_viewer_las(path: Path, xyz: np.ndarray, rgb: np.ndarray, classes: np.ndarray | None = None) -> None:
    """Write a LAS 1.4 (point format 7) file in the model frame for PotreeConverter."""
    header = laspy.LasHeader(point_format=7, version="1.4")
    header.scales = np.array([0.001, 0.001, 0.001])
    header.offsets = np.floor(xyz.min(axis=0)) if len(xyz) else np.zeros(3)
    las = laspy.LasData(header)
    las.x, las.y, las.z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    colours = np.asarray(rgb, dtype=np.uint16) * 257
    las.red, las.green, las.blue = colours[:, 0], colours[:, 1], colours[:, 2]
    if classes is not None:
        las.classification = np.clip(np.asarray(classes), 0, 255).astype(np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    las.write(path)


def _sparse_points(source: Path) -> tuple[np.ndarray, np.ndarray, None]:
    import pycolmap

    reconstruction = pycolmap.Reconstruction(source.parent)
    points = list(reconstruction.points3D.values())
    xyz = np.asarray([point.xyz for point in points], dtype=float).reshape(-1, 3)
    rgb = np.asarray([point.color for point in points], dtype=np.uint8).reshape(-1, 3)
    return xyz, rgb, None


def _ply_points(source: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    count, _properties, dtype, offsets = _read_ply_header(source)
    data = np.fromfile(source, dtype=dtype, count=count, offset=_ply_header_size(source))
    xyz = np.column_stack((data["x"], data["y"], data["z"])).astype(float)
    if {"red", "green", "blue"}.issubset(offsets):
        rgb = np.column_stack((data["red"], data["green"], data["blue"])).astype(np.uint8)
    else:
        rgb = np.full((count, 3), 200, np.uint8)
    classes = data["scalar_class"].astype(np.int64) if "scalar_class" in offsets else None
    return xyz, rgb, classes


def ensure_potree(project_directory: Path, kind: str) -> Path:
    """Convert the current sparse/dense cloud to a Potree octree once and reuse it until the source changes."""
    if kind not in {"sparse", "dense"}:
        raise ValueError("Unsupported point cloud kind")
    if not potree_available():
        raise FileNotFoundError("Potree or PotreeConverter is not installed")
    source = _source_path(project_directory, kind)
    if source is None:
        raise FileNotFoundError(f"No {kind} point cloud is available yet")
    viewer_root = project_directory / "products" / "viewer" / "potree"
    output = viewer_root / kind
    signature = {"source": str(source), "size": source.stat().st_size, "mtime_ns": source.stat().st_mtime_ns}
    signature_path = viewer_root / f"{kind}.json"
    if (output / "metadata.json").is_file() and signature_path.is_file():
        try:
            if json.loads(signature_path.read_text(encoding="utf-8")) == signature:
                return output
        except (OSError, ValueError):
            pass
    xyz, rgb, classes = _sparse_points(source) if kind == "sparse" else _ply_points(source)
    las_path = viewer_root / f"{kind}.las"
    write_viewer_las(las_path, xyz, rgb, classes)
    if output.exists():
        shutil.rmtree(output)
    result = subprocess.run(
        [str(POTREE_CONVERTER), str(las_path), "-o", str(output), "--method", "poisson"],
        capture_output=True, text=True, creationflags=_NO_CONSOLE,
    )
    if result.returncode != 0 or not (output / "metadata.json").is_file():
        raise RuntimeError(f"PotreeConverter failed: {(result.stderr or result.stdout)[-2000:]}")
    las_path.unlink(missing_ok=True)
    signature_path.write_text(json.dumps(signature), encoding="utf-8")
    return output
