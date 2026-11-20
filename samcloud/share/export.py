"""Build a static, self-contained Potree viewer package of a project's classified point cloud.

Pipeline (each step takes ``progress(fraction, message)`` and ``check()`` that raises ``ShareCancelled``):

1. ``prepare_points``  - load ``classified.ply`` (LAS class codes, confidence) and voxel-thin to ``max_points``.
2. ``build_octree``    - write a LAS file and convert it with the same PotreeConverter call the local viewer uses
   (``samcloud.viewer.potree``), attributes RGB + classification + intensity (= confidence).
3. ``assemble_package``- copy the slim viewer template and the Potree libraries, write legend and marker JSON.

The package contains only points, classes, control point names/positions and the project name; ``scan_package``
verifies that (no images, no absolute paths, no GPS reference data).
"""

from __future__ import annotations

import json
import logging
import re
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import laspy
import numpy as np
import yaml

from samcloud.core.paths import CONFIG_DIRECTORY, PROJECT_ROOT
from samcloud.drone.contracts import PointSet

LOGGER = logging.getLogger(__name__)

SHARE_CONFIG_PATH = CONFIG_DIRECTORY / "share.yaml"
TEMPLATE_DIRECTORY = Path(__file__).resolve().parent / "viewer_template"
POTREE_ASSETS = PROJECT_ROOT / "assets" / "potree"
DEFAULT_MAX_POINTS = 20_000_000
OCTREE_DIRECTORY = "pointcloud"
_HEX = re.compile(r"^#[0-9A-Fa-f]{6}$")

ProgressFn = Callable[[float, str], None]
CheckFn = Callable[[], None]

# (source below assets/potree, destination below the package)
_LIBRARY_FILES: list[tuple[str, str]] = [
    ("build/potree/potree.js", "lib/potree/potree.js"),
    ("build/potree/potree.css", "lib/potree/potree.css"),
    ("build/potree/LICENSE", "lib/potree/LICENSE"),
    ("libs/jquery/jquery-3.1.1.min.js", "lib/jquery/jquery-3.1.1.min.js"),
    ("libs/jquery-ui/jquery-ui.min.js", "lib/jquery-ui/jquery-ui.min.js"),
    ("libs/spectrum/spectrum.js", "lib/spectrum/spectrum.js"),
    ("libs/other/BinaryHeap.js", "lib/other/BinaryHeap.js"),
    ("libs/tween/tween.min.js", "lib/tween/tween.min.js"),
    ("libs/d3/d3.min.js", "lib/d3/d3.min.js"),
    ("libs/proj4/proj4.js", "lib/proj4/proj4.js"),
    ("libs/i18next/i18next.js", "lib/i18next/i18next.js"),
    ("libs/three.js/build/three.min.js", "lib/three/three.min.js"),
]
_LIBRARY_DIRECTORIES: list[tuple[str, str]] = [
    ("build/potree/workers/2.0", "lib/potree/workers/2.0"),
    ("build/potree/resources/lang", "lib/potree/resources/lang"),
    ("build/potree/resources/icons", "lib/potree/resources/icons"),
    ("build/potree/resources/images", "lib/potree/resources/images"),
    ("build/potree/resources/textures", "lib/potree/resources/textures"),
]
RESOURCE_PREFIX = "lib/potree/resources/"  # Potree UI icons/textures: the only images a package may contain


class ShareCancelled(Exception):
    """The user cancelled the export."""


def load_share_config(path: Path | str | None = None) -> dict[str, Any]:
    """Read ``config/share.yaml`` (defaults for missing keys)."""
    config: dict[str, Any] = {"provider": "cloudflare_r2", "expiry_days": 15, "max_points": DEFAULT_MAX_POINTS, "octree_encoding": "BROTLI",
                              "providers": {}}
    source = Path(path) if path else SHARE_CONFIG_PATH
    try:
        loaded = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError) as error:
        LOGGER.warning("share config unreadable (%s); using defaults", error)
        loaded = {}
    if isinstance(loaded, dict):
        config.update(loaded)
    config["max_points"] = max(1000, int(config.get("max_points") or DEFAULT_MAX_POINTS))
    return config


# ------------------------------------------------------------------------------------------------
# 1. points
# ------------------------------------------------------------------------------------------------

def _voxel_keys(xyz: np.ndarray, origin: np.ndarray, size: float, class_index: np.ndarray, n_classes: int) -> np.ndarray:
    cells = np.floor((xyz - origin) / size).astype(np.int64)
    dims = cells.max(axis=0) + 1
    key = (cells[:, 0] * dims[1] + cells[:, 1]) * dims[2] + cells[:, 2]
    return key * n_classes + class_index


def _first_per_key(keys: np.ndarray) -> np.ndarray:
    return np.unique(keys, return_index=True)[1]


def _thin_indices(xyz: np.ndarray, target: int, check: CheckFn | None) -> np.ndarray:
    """Indices of at most ``target`` spatially evenly spread points (voxel grid found by bisection)."""
    n = len(xyz)
    if n <= target:
        return np.arange(n)
    origin = xyz.min(axis=0)
    extent = np.maximum(xyz.max(axis=0) - origin, 1e-6)
    zeros = np.zeros(n, dtype=np.int64)
    # smallest size whose cell index still fits into int64
    low = float((np.prod(extent + 1.0) / 2.0**61) ** (1.0 / 3.0)) + 1e-9
    high = float(extent.max()) * 2.0
    best: np.ndarray | None = None
    for _ in range(16):
        if check:
            check()
        size = float(np.sqrt(low * high))
        selected = _first_per_key(_voxel_keys(xyz, origin, size, zeros, 1))
        if len(selected) <= target:
            best, high = selected, size
            if len(selected) >= 0.92 * target:
                break
        else:
            low = size
    if best is None:
        best = _first_per_key(_voxel_keys(xyz, origin, high, zeros, 1))
    if len(best) > target:  # only for degenerate data (all points in one place)
        best = np.random.default_rng(0).choice(best, size=target, replace=False)
    return best


def thin_points(points: PointSet, max_points: int, *, check: CheckFn | None = None) -> PointSet:
    """Voxel-thin ``points`` to at most ``max_points`` keeping every class and its share of the cloud.

    Every class is thinned on its own voxel grid to its proportional share of ``max_points`` (at least one point
    per class), so class proportions stay exact and sparse classes do not vanish. Returns ``points`` unchanged
    when it is small enough. Kept rows retain colour, class and confidence.
    """
    n = len(points)
    max_points = int(max_points)
    if n <= max_points:
        return points
    codes, inverse, counts = np.unique(points.classes, return_inverse=True, return_counts=True)
    inverse = inverse.reshape(-1)
    budget = max(max_points - len(codes), 1)
    kept: list[np.ndarray] = []
    for index in range(len(codes)):
        rows = np.flatnonzero(inverse == index)
        target = max(int(budget * counts[index] // n), 1)
        kept.append(rows[_thin_indices(points.xyz[rows], target, check)])
    selected = np.sort(np.concatenate(kept))
    if len(selected) > max_points:
        selected = np.sort(np.random.default_rng(0).choice(selected, size=max_points, replace=False))
    return points.subset(selected)


def prepare_points(project_dir: Path, max_points: int, progress: ProgressFn, check: CheckFn) -> tuple[PointSet, int]:
    """Load the classified cloud of the project and thin it; returns ``(points, original_count)``."""
    from samcloud.drone.io import load_classified_pointset

    project_dir = Path(project_dir)
    class_set = _class_set(project_dir)
    progress(0.0, "Punktwolke laden")
    points = load_classified_pointset(project_dir / "products", class_set=class_set)
    check()
    original = len(points)
    progress(0.5, f"{original:,} Punkte, Ausdünnen".replace(",", "'"))
    points = thin_points(points, max_points, check=check)
    progress(1.0, f"{len(points):,} Punkte".replace(",", "'"))
    return points, original


def _class_set(project_dir: Path) -> str:
    project = _load_project(project_dir)
    return str(project.get("settings", {}).get("classification", {}).get("class_set", "outdoor"))


def _load_project(project_dir: Path) -> dict[str, Any]:
    path = Path(project_dir) / "project.samcloud.yml"
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError):
        return {}
    return data if isinstance(data, dict) else {}


# ------------------------------------------------------------------------------------------------
# 2. octree
# ------------------------------------------------------------------------------------------------

def write_share_las(path: Path, points: PointSet) -> None:
    """Write a LAS 1.4 (format 7) file: RGB, classification (LAS code), intensity = confidence (0..65535)."""
    xyz = points.xyz
    header = laspy.LasHeader(point_format=7, version="1.4")
    header.scales = np.array([0.001, 0.001, 0.001])
    header.offsets = np.floor(xyz.min(axis=0)) if len(xyz) else np.zeros(3)
    las = laspy.LasData(header)
    las.x, las.y, las.z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    rgb = points.rgb if points.rgb is not None else np.full((len(xyz), 3), 200, np.uint8)
    colours = np.asarray(rgb, dtype=np.uint16) * 257
    las.red, las.green, las.blue = colours[:, 0], colours[:, 1], colours[:, 2]
    las.classification = np.clip(np.asarray(points.classes), 0, 255).astype(np.uint8)
    if points.confidence is not None:
        las.intensity = np.clip(np.asarray(points.confidence, dtype=np.float64) * 65535.0, 0, 65535).astype(np.uint16)
    path.parent.mkdir(parents=True, exist_ok=True)
    las.write(path)


def _prune_octree(octree_dir: Path) -> None:
    """Keep only the files Potree reads (metadata.json, hierarchy.bin, octree.bin); drop logs and stage data."""
    keep = {"metadata.json", "hierarchy.bin", "octree.bin"}
    if not octree_dir.is_dir():
        return
    for entry in list(octree_dir.iterdir()):
        if entry.name in keep:
            continue
        if entry.is_dir():
            shutil.rmtree(entry, ignore_errors=True)
        else:
            entry.unlink(missing_ok=True)


def build_octree(points: PointSet, work_dir: Path, octree_dir: Path, progress: ProgressFn, check: CheckFn,
                 *, encoding: str = "BROTLI", converter: Path | None = None) -> dict[str, Any]:
    """Convert ``points`` to a Potree octree in ``octree_dir`` with the converter of the local viewer."""
    from samcloud.viewer import potree as local_potree

    exe = Path(converter) if converter else local_potree.POTREE_CONVERTER
    if not exe.is_file():
        raise FileNotFoundError("PotreeConverter is not installed")
    work_dir.mkdir(parents=True, exist_ok=True)
    las_path = work_dir / "share.las"
    progress(0.0, "Punkte schreiben")
    write_share_las(las_path, points)
    check()
    if octree_dir.exists():
        shutil.rmtree(octree_dir)
    log_path = work_dir / "converter.log"
    command = [str(exe), str(las_path), "-o", str(octree_dir), "--method", "poisson", "--encoding", encoding,
               "--attributes", "rgb", "classification", "intensity"]
    started = time.monotonic()
    progress(0.1, "Octree bauen")
    with log_path.open("wb") as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, creationflags=local_potree._NO_CONSOLE)
        try:
            while process.poll() is None:
                try:
                    check()
                except ShareCancelled:
                    process.kill()
                    process.wait()
                    raise
                elapsed = time.monotonic() - started
                progress(0.1 + 0.85 * (1.0 - 1.0 / (1.0 + elapsed / 60.0)), "Octree bauen")
                time.sleep(0.25)
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()
    las_path.unlink(missing_ok=True)
    if (octree_dir / "log.txt").is_file():  # the converter log names local paths; it must not be published
        shutil.copyfile(octree_dir / "log.txt", log_path)
    _prune_octree(octree_dir)
    if process.returncode != 0 or not (octree_dir / "metadata.json").is_file():
        tail = log_path.read_text(encoding="utf-8", errors="replace")[-1500:]
        raise RuntimeError(f"PotreeConverter fehlgeschlagen: {tail}")
    progress(1.0, "Octree fertig")
    return json.loads((octree_dir / "metadata.json").read_text(encoding="utf-8"))


# ------------------------------------------------------------------------------------------------
# 3. package
# ------------------------------------------------------------------------------------------------

def build_legend(project_dir: Path, classes: np.ndarray) -> list[dict[str, Any]]:
    """Legend entries (LAS code, German name, colour, point count) of the classes present in ``classes``."""
    from samcloud.classification.custom import load_project_classes
    from samcloud.classification.service import class_default_colors, class_las_codes

    project = _load_project(project_dir)
    class_set = str(project.get("settings", {}).get("classification", {}).get("class_set", "outdoor"))
    config = load_project_classes(project_dir)
    codes = class_las_codes(class_set, config)
    names = list(config.get("labels_de", {}).get(class_set) or [])
    defaults = class_default_colors(class_set, config) or ["#B0B0B0"] * len(codes)
    saved = project.get("settings", {}).get("classification", {}).get("class_colors", {})
    saved = saved if isinstance(saved, dict) else {}
    values, counts = np.unique(np.asarray(classes), return_counts=True)
    present = {int(v): int(c) for v, c in zip(values, counts)}
    entries: list[dict[str, Any]] = []
    used: set[int] = set()
    for index, code in enumerate(codes):
        code = int(code)
        if code in used or code not in present:
            continue
        used.add(code)
        colour = saved.get(str(index), defaults[index])
        entries.append({
            "code": code,
            "name": str(names[index]) if index < len(names) and names[index] else f"Klasse {code}",
            "color": str(colour).upper() if isinstance(colour, str) and _HEX.match(colour) else defaults[index],
            "count": present[code],
        })
    for code, count in sorted(present.items()):
        if code not in used:
            entries.append({"code": code, "name": f"Klasse {code}", "color": "#B0B0B0", "count": count})
    return sorted(entries, key=lambda item: item["code"])


def collect_markers(project_dir: Path) -> list[dict[str, Any]]:
    """Control point / AprilTag markers of the dense viewer, reduced to name, kind, position and colour."""
    try:
        from samcloud.app.potree_viewer import control_point_markers

        raw = control_point_markers(project_dir)
    except Exception as error:  # markers are optional
        LOGGER.warning("share: markers unavailable: %s", error)
        return []
    markers: list[dict[str, Any]] = []
    for item in raw:
        coordinates = item.get("coordinates")
        if not (isinstance(coordinates, (list, tuple)) and len(coordinates) == 3):
            continue
        entry: dict[str, Any] = {
            "name": str(item.get("name", "")),
            "kind": str(item.get("kind", "passpunkt")),
            "state_color": str(item.get("state_color", item.get("kind", "passpunkt"))),
            "coordinates": [float(v) for v in coordinates],
        }
        corners = item.get("corners")
        if isinstance(corners, (list, tuple)) and len(corners) == 4:
            try:
                entry["corners"] = [[float(v) for v in corner] for corner in corners]
            except (TypeError, ValueError):
                pass
        markers.append(entry)
    return markers


def _copy_tree(source: Path, destination: Path) -> None:
    for path in source.rglob("*"):
        if path.is_file():
            target = destination / path.relative_to(source)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)


def assemble_package(project_dir: Path, package_dir: Path, *, points: PointSet, original_points: int,
                     metadata: dict[str, Any], progress: ProgressFn | None = None) -> dict[str, Any]:
    """Fill ``package_dir`` (octree already in ``package_dir/pointcloud``) with viewer, libraries and JSON data."""
    progress = progress or (lambda fraction, message: None)
    progress(0.0, "Viewer kopieren")
    _copy_tree(TEMPLATE_DIRECTORY, package_dir)
    for source, destination in _LIBRARY_FILES:
        target = package_dir / destination
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(POTREE_ASSETS / source, target)
    for source, destination in _LIBRARY_DIRECTORIES:
        _copy_tree(POTREE_ASSETS / source, package_dir / destination)
    progress(0.6, "Legende und Marker")
    project = _load_project(project_dir)
    name = re.sub(r"[\x00-\x1f]", " ", str(project.get("project", {}).get("name") or "SAMcloud"))[:120]
    legend = {
        "project_name": name,
        "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "points": len(points),
        "source_points": int(original_points),
        "generator": "SAMcloud",
        "bounds": metadata.get("boundingBox"),
        "classes": build_legend(project_dir, points.classes),
    }
    markers = collect_markers(project_dir)
    data = package_dir / "data"
    data.mkdir(parents=True, exist_ok=True)
    (data / "legend.json").write_text(json.dumps(legend, ensure_ascii=False, indent=1), encoding="utf-8")
    (data / "markers.json").write_text(json.dumps({"markers": markers}, ensure_ascii=False, indent=1), encoding="utf-8")
    progress(1.0, "Paket fertig")
    return legend


def package_size_mb(package_dir: Path) -> float:
    """Total package size in megabytes."""
    return sum(p.stat().st_size for p in Path(package_dir).rglob("*") if p.is_file()) / 1e6


# ------------------------------------------------------------------------------------------------
# privacy
# ------------------------------------------------------------------------------------------------

_FORBIDDEN_SUFFIXES = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tif", ".tiff", ".heic", ".webp"}
_TEXT_SUFFIXES = {".html", ".js", ".css", ".json", ".txt", ".yml", ".yaml", ".md", ".svg"}
_WINDOWS_PATH = re.compile(r"(?<![A-Za-z0-9])[A-Za-z]:[\\/]{1,2}[A-Za-z0-9_.$ -]+[\\/]")
_UNIX_HOME = re.compile(r"(?:/Users|/home)/[A-Za-z0-9_.-]+/")


def scan_package(package_dir: Path, *, allowed_image_dirs: tuple[str, ...] = (RESOURCE_PREFIX,)) -> list[str]:
    """Return privacy violations of a package: images, absolute local paths, ``gps_ref`` data (empty = clean)."""
    root = Path(package_dir)
    problems: list[str] = []
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        relative = path.relative_to(root).as_posix()
        suffix = path.suffix.lower()
        if suffix in _FORBIDDEN_SUFFIXES and not any(relative.startswith(prefix) for prefix in allowed_image_dirs):
            problems.append(f"image file: {relative}")
            continue
        if path.name.lower().startswith("gps_ref") or path.name.lower() == "georeference.json":
            problems.append(f"gps data file: {relative}")
            continue
        if suffix in _TEXT_SUFFIXES:
            text = path.read_text(encoding="utf-8", errors="replace")
            if "gps_ref" in text:
                problems.append(f"gps_ref reference: {relative}")
            if relative.startswith("lib/"):  # third-party libraries: URLs in comments are not local paths
                continue
            if _WINDOWS_PATH.search(text) or _UNIX_HOME.search(text):
                problems.append(f"absolute path: {relative}")
    return problems
