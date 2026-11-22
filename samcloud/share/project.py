"""Build the static package for a complete project share."""

from __future__ import annotations

import json
import logging
import re
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import ezdxf
from ezdxf import colors
from ezdxf.disassemble import recursive_decompose
from ezdxf.path import make_path

from samcloud.core.paths import PROJECT_ROOT
from samcloud.share import export

LOGGER = logging.getLogger(__name__)

PROJECT_TEMPLATE_DIRECTORY = Path(__file__).resolve().parent / "project_template"
THREE_ASSETS = PROJECT_ROOT / "assets" / "vendor" / "three"
_THREE_FILES = ("three.module.js", "OrbitControls.js", "GLTFLoader.js", "BufferGeometryUtils.js", "LICENSE")


@dataclass(frozen=True)
class Level:
    """The optional DXF and IFC views of one project level."""

    key: str
    label: str
    dxf: Path | None = None
    ifc_dir: Path | None = None


def _project_level_names(project_dir: Path) -> dict[str, str]:
    """Return level id to human name from durable DXF state and project metadata."""
    names: dict[str, str] = {}
    try:
        from samcloud.dxf.state import load_state

        for level in load_state(project_dir).levels:
            if isinstance(level, dict) and level.get("id") is not None and level.get("name"):
                names[str(level["id"])] = str(level["name"])
    except Exception as error:  # state is optional for a share
        LOGGER.debug("project share: DXF state unavailable: %s", error)
    document = export._load_project(project_dir)
    for source in (document.get("levels"), document.get("dxf", {}).get("levels") if isinstance(document.get("dxf"), dict) else None):
        if isinstance(source, list):
            for level in source:
                if isinstance(level, dict) and level.get("id") is not None and level.get("name"):
                    names.setdefault(str(level["id"]), str(level["name"]))
    return names


def _file_suffix(path: Path, project_name: str) -> str:
    stem = path.stem
    if stem.lower().endswith("_clean"):
        stem = stem[:-6]
    prefix = f"{project_name}_"
    return stem[len(prefix):] if stem.lower().startswith(prefix.lower()) else stem


def _safe_key(value: str, fallback: str) -> str:
    key = re.sub(r"[^a-z0-9_]+", "_", value.lower()).strip("_")
    return key or fallback


def find_project_outputs(project_dir: Path | str) -> list[Level]:
    """Find and pair export DXFs with viewable IFC/GLB folders."""
    directory = Path(project_dir)
    names = _project_level_names(directory)
    by_suffix = {"site": "site"}
    for key, label in names.items():
        by_suffix[_safe_key(label, key)] = key
        by_suffix[_safe_key(key, key)] = key

    ifc_dirs: dict[str, Path] = {}
    bim = directory / "products" / "bim"
    if bim.is_dir():
        for candidate in sorted(bim.iterdir()):
            if candidate.is_dir() and (candidate / "model.glb").is_file():
                ifc_dirs[candidate.name] = candidate

    dxfs: dict[str, Path] = {}
    unmatched_dxfs: list[tuple[str, Path]] = []
    export_dir = directory / "products" / "dxf" / "export"
    for path in sorted(export_dir.glob("*_clean.dxf")) if export_dir.is_dir() else []:
        suffix = _file_suffix(path, directory.name)
        normalized = _safe_key(suffix, "dxf")
        key = "site" if normalized == "site" else by_suffix.get(normalized)
        if key is None:
            # A corresponding IFC key is a useful fallback when no DXF state was saved.
            key = next((item for item in ifc_dirs if _safe_key(item, item) == normalized), None)
        if key is None:
            unmatched_dxfs.append((normalized, path))
        else:
            dxfs[key] = path
    free_ifc_keys = [key for key in ifc_dirs if key not in dxfs]
    for normalized, path in unmatched_dxfs:
        if free_ifc_keys:
            dxfs[free_ifc_keys.pop(0)] = path
        else:
            dxfs[f"dxf_{normalized}"] = path

    keys: list[str] = []
    for key in ifc_dirs:
        if key not in keys:
            keys.append(key)
    for key in dxfs:
        if key not in keys:
            keys.append(key)
    keys.sort(key=lambda key: (key != "site", key))
    return [Level(key, names.get(key, "Site" if key == "site" else key), dxfs.get(key), ifc_dirs.get(key)) for key in keys]


def project_share_available(project_dir: Path | str, project: dict[str, Any] | None = None) -> bool:
    """True when all three project views have their required source artefacts."""
    directory = Path(project_dir)
    if not (directory / "products" / "classified" / "classified.ply").is_file():
        return False
    outputs = find_project_outputs(directory)
    return any(level.dxf is not None for level in outputs) and any(level.ifc_dir is not None for level in outputs)


def _layer_colour(layer: Any) -> str:
    value = getattr(layer.dxf, "true_color", None)
    if value is not None:
        red, green, blue = colors.int2rgb(int(value))
    else:
        red, green, blue = colors.aci2rgb(abs(int(layer.dxf.color)))
    return f"#{red:02x}{green:02x}{blue:02x}"


def _text_item(entity: Any) -> tuple[float, float, float, float, str] | None:
    try:
        kind = entity.dxftype()
        if kind == "MTEXT":
            point = entity.dxf.insert
            text = entity.plain_text()
            height = float(entity.dxf.char_height)
            rotation = float(getattr(entity.dxf, "rotation", 0.0))
        else:
            point = entity.dxf.insert
            text = str(entity.dxf.text)
            height = float(entity.dxf.height)
            rotation = float(getattr(entity.dxf, "rotation", 0.0))
        return float(point.x), float(point.y), height, rotation, text
    except Exception as error:
        LOGGER.debug("project share: text skipped: %s", error)
        return None


def dxf_to_json(dxf_path: Path | str, *, flatten_distance: float = 0.01) -> dict[str, Any]:
    """Convert a DXF modelspace to compact, origin-relative 2D JSON."""
    document = ezdxf.readfile(str(dxf_path))
    records: dict[str, dict[str, list[Any]]] = {}

    def record(name: str) -> dict[str, list[Any]]:
        return records.setdefault(name, {"polylines": [], "texts": []})

    for entity in recursive_decompose(document.modelspace()):
        try:
            kind = entity.dxftype()
            layer_name = str(entity.dxf.layer)
            if kind in {"TEXT", "MTEXT", "ATTRIB"}:
                item = _text_item(entity)
                if item is not None:
                    record(layer_name)["texts"].append(item)
                continue
            if kind in {"3DSOLID", "MESH", "POLYFACE", "ACISOLID", "BODY", "REGION"}:
                continue
            path = make_path(entity)
            points = [(float(point.x), float(point.y)) for point in path.flattening(flatten_distance)]
            if len(points) >= 2:
                record(layer_name)["polylines"].append(points)
        except Exception as error:
            LOGGER.debug("project share: DXF entity skipped: %s", error)

    all_points = [point for data in records.values() for line in data["polylines"] for point in line]
    all_points.extend((item[0], item[1]) for data in records.values() for item in data["texts"])
    min_x = min((point[0] for point in all_points), default=0.0)
    min_y = min((point[1] for point in all_points), default=0.0)
    max_x = max((point[0] for point in all_points), default=min_x)
    max_y = max((point[1] for point in all_points), default=min_y)

    layers: list[dict[str, Any]] = []
    for name in sorted(records):
        data = records[name]
        if not data["polylines"] and not data["texts"]:
            continue
        try:
            layer = document.layers.get(name)
            visible = int(layer.dxf.color) >= 0 and not bool(getattr(layer.dxf, "flags", 0) & 1)
            colour = _layer_colour(layer)
        except Exception:
            visible, colour = True, "#ffffff"
        polylines = [[coordinate for x, y in line for coordinate in
                      (int(round((x - min_x) * 1000)), int(round((y - min_y) * 1000)))]
                     for line in data["polylines"]]
        texts = [{"x": int(round((x - min_x) * 1000)), "y": int(round((y - min_y) * 1000)),
                  "h": int(round(height * 1000)), "rot": rotation, "s": text}
                 for x, y, height, rotation, text in data["texts"]]
        layers.append({"name": name, "color": colour, "visible": visible, "polylines": polylines, "texts": texts})
    result = {"version": 1, "name": Path(dxf_path).name, "units": "m", "scale": 0.001,
              "origin": [min_x, min_y], "bounds": [0, 0, int(round((max_x - min_x) * 1000)), int(round((max_y - min_y) * 1000))],
              "background": "#0d1b2a", "layers": layers}
    if len(json.dumps(result, ensure_ascii=False, separators=(",", ":")).encode("utf-8")) > 30 * 1024 * 1024:
        LOGGER.warning("project share: DXF JSON exceeds 30 MB: %s", dxf_path)
    return result


def _category_labels(language: str) -> dict[str, str]:
    path = PROJECT_ROOT / "samcloud" / "app" / "locales" / language / "ifc.json"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        category = data.get("category") if isinstance(data, dict) else None
        return {str(key): str(value) for key, value in category.items()} if isinstance(category, dict) else {}
    except (OSError, ValueError):
        return {}


def ifc_view_json(level_key: str, ifc_dir: Path | str, lang: str) -> dict[str, Any]:
    """Return the small viewer configuration derived from an IFC props.json file."""
    from samcloud.bim import ui_data

    directory = Path(ifc_dir)
    try:
        data = json.loads((directory / "props.json").read_text(encoding="utf-8"))
        data = data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        data = {}
    cfg = ui_data.viewer_config(level_key, data)
    counts = dict(cfg.get("counts", {}))
    outdoor = bool(cfg.get("outdoor"))
    order = [name for name in ui_data.OUTDOOR_CATEGORY_ORDER if name in counts] if outdoor else sorted(counts, key=lambda name: (-int(counts[name]), name))
    translations = _category_labels(lang)
    names = _project_level_names(directory.parents[2]) if len(directory.parents) >= 3 else {}
    label = names.get(level_key, "Site" if level_key == "site" else level_key)
    return {"key": level_key, "label": label, "outdoor": outdoor, "order": order,
            "labels": {name: translations.get(name, name) for name in order}, "counts": counts,
            "hidden": sorted(cfg.get("hidden", set())), "colors": dict(cfg.get("colors", {})),
            "by_node": dict(cfg.get("by_node", {}))}


def _copy_tree(source: Path, destination: Path) -> None:
    for path in source.rglob("*"):
        if path.is_file():
            target = destination / path.relative_to(source)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)


def build_project_package(project_dir: Path | str, package_dir: Path | str, *, config: dict[str, Any],
                          progress_for_phase: Callable[[int], Callable[[float, str], None]], check: Callable[[], None],
                          max_points: int, converter: Path | None = None) -> dict[str, Any]:
    """Build the point cloud, DXF and IFC views for a project-share package."""
    directory, package = Path(project_dir), Path(package_dir)
    prepare = progress_for_phase(0)
    points, original = export.prepare_points(directory, max_points, prepare, check)
    check()
    octree = progress_for_phase(1)
    metadata = export.build_octree(points, package.parent / "work", package / "pc" / export.OCTREE_DIRECTORY, octree, check,
                                   encoding=str(config.get("octree_encoding", "BROTLI")), converter=converter)
    levels = find_project_outputs(directory)

    dxf_progress = progress_for_phase(2)
    dxf_levels = [level for level in levels if level.dxf is not None]
    for index, level in enumerate(dxf_levels, start=1):
        check()
        target = package / "data" / "dxf" / f"{level.key}.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(dxf_to_json(level.dxf), ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
        dxf_progress(index / len(dxf_levels), "DXF aufbereiten")
    if not dxf_levels:
        dxf_progress(1.0, "DXF aufbereiten")

    try:
        from samcloud.app.i18n import current_language
        language = current_language()
    except Exception:
        language = "de"
    language = language if language in {"de", "en"} else "de"
    ifc_progress = progress_for_phase(3)
    ifc_levels = [level for level in levels if level.ifc_dir is not None]
    for index, level in enumerate(ifc_levels, start=1):
        check()
        target = package / "data" / "ifc" / level.key
        target.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(level.ifc_dir / "model.glb", target / "model.glb")
        (target / "view.json").write_text(json.dumps(ifc_view_json(level.key, level.ifc_dir, language), ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
        ifc_progress(index / len(ifc_levels), "IFC aufbereiten")
    if not ifc_levels:
        ifc_progress(1.0, "IFC aufbereiten")

    assemble = progress_for_phase(4)
    export.assemble_package(directory, package / "pc", points=points, original_points=original, metadata=metadata, progress=assemble, register=True)
    _copy_tree(PROJECT_TEMPLATE_DIRECTORY, package)
    for name in _THREE_FILES:
        target = package / "lib" / "three" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(THREE_ASSETS / name, target)
    document = export._load_project(directory)
    name = str(document.get("project", {}).get("name") or directory.name)
    view_names = [name for name, present in (("dxf", dxf_levels), ("pc", True), ("ifc", ifc_levels)) if present]
    level_rows = [{"key": level.key, "label": level.label,
                   "dxf": f"data/dxf/{level.key}.json" if level.dxf else None,
                   "ifc": f"data/ifc/{level.key}/" if level.ifc_dir else None} for level in levels]
    project_data = {"version": 1, "name": name, "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
                    "lang": language, "generator": "SAMcloud", "points": len(points), "views": view_names,
                    "default_level": next((level.key for level in levels if level.key == "site"), levels[0].key if levels else "site"),
                    "levels": level_rows}
    (package / "data" / "project.json").write_text(json.dumps(project_data, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    problems = export.scan_package(package)
    if problems:
        raise RuntimeError("Datenschutzprüfung fehlgeschlagen: " + "; ".join(problems[:5]))
    return {"points": len(points), "views": view_names, "levels": [level.key for level in levels]}
