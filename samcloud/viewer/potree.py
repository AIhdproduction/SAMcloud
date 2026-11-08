"""Create the class scheme consumed by the Potree viewer integration."""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
from threading import Lock, Thread
import time
from typing import Any

import laspy
import numpy as np

from samcloud.core.paths import PROJECT_ROOT
from samcloud.viewer.preview import _ply_header_size, _read_ply_header, _source_path
from samcloud.viewer.sparse_filter import FILTER_VERSION, load_clean_sparse_points, sparse_cleaning_settings

POTREE_CONVERTER = PROJECT_ROOT / "tools" / "PotreeConverter" / "PotreeConverter.exe"
POTREE_LIBRARY = PROJECT_ROOT / "assets" / "potree" / "build" / "potree" / "potree.js"
_NO_CONSOLE = 0x08000000 if os.name == "nt" else 0
_POTREE_LOCKS: dict[tuple[Path, str], Lock] = {}
_POTREE_LOCKS_GUARD = Lock()
_POTREE_BUILD_STATES: dict[tuple[Path, str], dict[str, object]] = {}
_POTREE_BUILD_STATES_GUARD = Lock()


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
    """Sparse tie points for display, with stray points removed (model stays untouched)."""
    xyz, rgb = load_clean_sparse_points(source)
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


def _conversion_lock(project_directory: Path, kind: str) -> Lock:
    """Return the shared converter lock for one project cloud."""
    key = (project_directory.resolve(), kind)
    with _POTREE_LOCKS_GUARD:
        return _POTREE_LOCKS.setdefault(key, Lock())


def _cache_matches(output: Path, signature_path: Path, signature: dict[str, object]) -> bool:
    if not (output / "metadata.json").is_file() or not signature_path.is_file():
        return False
    try:
        return json.loads(signature_path.read_text(encoding="utf-8")) == signature
    except (OSError, ValueError):
        return False


def _potree_paths(project_directory: Path, kind: str) -> tuple[Path, Path, Path, dict[str, object]]:
    """Return the source, output, signature path, and current cache signature."""
    source = _source_path(project_directory, kind)
    if source is None:
        raise FileNotFoundError(f"No {kind} point cloud is available yet")
    viewer_root = project_directory / "products" / "viewer" / "potree"
    output = viewer_root / kind
    signature = {
        "source": str(source), "size": source.stat().st_size, "mtime_ns": source.stat().st_mtime_ns,
        "encoding": "BROTLI",
    }
    if kind == "sparse":
        signature["sparse_filter"] = {"version": FILTER_VERSION, **sparse_cleaning_settings()}
    return source, output, viewer_root / f"{kind}.json", signature


def _build_key(project_directory: Path, kind: str) -> tuple[Path, str]:
    return (project_directory.resolve(), kind)


def potree_status(project_directory: Path, kind: str) -> dict[str, object]:
    """Report whether a project's Potree octree is ready, building, or unavailable."""
    if kind not in {"sparse", "dense"}:
        raise ValueError("Unsupported point cloud kind")
    project_directory = Path(project_directory).resolve()
    if not potree_available():
        return {"state": "error", "message": "Potree or PotreeConverter is not installed", "started_at": None}
    try:
        _source, output, signature_path, signature = _potree_paths(project_directory, kind)
    except FileNotFoundError as error:
        return {"state": "missing", "message": str(error), "started_at": None}
    except OSError as error:
        return {"state": "error", "message": str(error), "started_at": None}
    if _cache_matches(output, signature_path, signature):
        return {"state": "ready", "message": "", "started_at": None}
    with _POTREE_BUILD_STATES_GUARD:
        state = _POTREE_BUILD_STATES.get(_build_key(project_directory, kind))
        if state is not None:
            return dict(state)
    return {"state": "missing", "message": "Point cloud has not been prepared yet", "started_at": None}


def start_potree_build(project_directory: Path, kind: str) -> None:
    """Start one background Potree conversion when the current octree is not ready."""
    if kind not in {"sparse", "dense"}:
        raise ValueError("Unsupported point cloud kind")
    project_directory = Path(project_directory).resolve()
    if not potree_available():
        return
    try:
        _potree_paths(project_directory, kind)
    except OSError:
        return
    if potree_status(project_directory, kind)["state"] == "ready":
        return
    key = _build_key(project_directory, kind)
    with _POTREE_BUILD_STATES_GUARD:
        current = _POTREE_BUILD_STATES.get(key)
        if current is not None and current["state"] == "building":
            return
        _POTREE_BUILD_STATES[key] = {
            "state": "building",
            "message": "Preparing point cloud",
            "started_at": time.time(),
        }

    def build() -> None:
        try:
            ensure_potree(project_directory, kind)
        except Exception as error:
            state = {"state": "error", "message": str(error), "started_at": None}
        else:
            state = {"state": "ready", "message": "", "started_at": None}
        with _POTREE_BUILD_STATES_GUARD:
            _POTREE_BUILD_STATES[key] = state

    Thread(target=build, name=f"potree-{kind}", daemon=True).start()


def ensure_potree(project_directory: Path, kind: str) -> Path:
    """Convert the current sparse/dense cloud to a Potree octree once and reuse it until the source changes."""
    if kind not in {"sparse", "dense"}:
        raise ValueError("Unsupported point cloud kind")
    if not potree_available():
        raise FileNotFoundError("Potree or PotreeConverter is not installed")
    project_directory = Path(project_directory).resolve()
    source, output, signature_path, signature = _potree_paths(project_directory, kind)
    if _cache_matches(output, signature_path, signature):
        return output
    with _conversion_lock(project_directory, kind):
        source, output, signature_path, signature = _potree_paths(project_directory, kind)
        if _cache_matches(output, signature_path, signature):
            return output
        xyz, rgb, classes = _sparse_points(source) if kind == "sparse" else _ply_points(source)
        las_path = signature_path.parent / f"{kind}.las"
        write_viewer_las(las_path, xyz, rgb, classes)
        if output.exists():
            shutil.rmtree(output)
        result = subprocess.run(
            [str(POTREE_CONVERTER), str(las_path), "-o", str(output), "--method", "poisson", "--encoding", "BROTLI"],
            capture_output=True, text=True, creationflags=_NO_CONSOLE,
        )
        if result.returncode != 0 or not (output / "metadata.json").is_file():
            raise RuntimeError(f"PotreeConverter failed: {(result.stderr or result.stdout)[-2000:]}")
        las_path.unlink(missing_ok=True)
        signature_path.write_text(json.dumps(signature), encoding="utf-8")
        return output
