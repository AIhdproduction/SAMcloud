"""Read-only access to the BIM viewer artefacts of a project (IFC, GLB, props.json, qa.json).

Layout: ``<project>/products/bim/<level_key>/`` with ``<project_name>_<level_key>.ifc``, ``model.glb``,
``props.json`` and ``qa.json``. The level key is the one the DXF page uses ("1", "site", ...).
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Literal

IfcStatus = Literal["missing", "fresh", "stale"]

GLB_NAME = "model.glb"
PROPS_NAME = "props.json"
QA_NAME = "qa.json"

_PROPS_CACHE: dict[Path, tuple[tuple[int, int], dict[str, Any]]] = {}


def bim_dir(project_root: Path | str, level_key: str) -> Path:
    """Folder holding the BIM artefacts of one level (may not exist)."""
    key = str(level_key)
    if not key or key in {".", ".."} or "/" in key or "\\" in key:
        raise ValueError(f"Invalid level key '{level_key}'")
    return Path(project_root) / "products" / "bim" / key


def find_ifc(project_root: Path | str, level_key: str) -> Path | None:
    """The IFC file of a level: ``props.json``'s ``ifc_file`` if present, else the newest ``*.ifc``."""
    folder = bim_dir(project_root, level_key)
    if not folder.is_dir():
        return None
    data = props(project_root, level_key)
    name = data.get("ifc_file") if data else None
    if isinstance(name, str) and name and Path(name).name == name:
        candidate = folder / name
        if candidate.is_file():
            return candidate
    matches = sorted(folder.glob("*.ifc"), key=lambda path: (path.stat().st_mtime, path.name))
    return matches[-1] if matches else None


def glb_path(project_root: Path | str, level_key: str) -> Path | None:
    """The viewer GLB of a level, or None."""
    path = bim_dir(project_root, level_key) / GLB_NAME
    return path if path.is_file() else None


def props_path(project_root: Path | str, level_key: str) -> Path | None:
    path = bim_dir(project_root, level_key) / PROPS_NAME
    return path if path.is_file() else None


def props(project_root: Path | str, level_key: str) -> dict[str, Any]:
    """Parsed props.json (cached by mtime and size); an empty dict when missing or unreadable."""
    path = props_path(project_root, level_key)
    if path is None:
        return {}
    try:
        stat = path.stat()
        signature = (stat.st_mtime_ns, stat.st_size)
        cached = _PROPS_CACHE.get(path)
        if cached is not None and cached[0] == signature:
            return cached[1]
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    if not isinstance(data, dict):
        return {}
    _PROPS_CACHE[path] = (signature, data)
    return data


def qa(project_root: Path | str, level_key: str) -> dict[str, Any]:
    """Parsed qa.json, or an empty dict when missing or unreadable."""
    path = bim_dir(project_root, level_key) / QA_NAME
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def plan_model_file(project_root: Path | str, level_key: str) -> Path:
    """Location of the ortho plan_model.json that the IFC is built from (read only, may not exist)."""
    from samcloud.ortho.plan import clean_plan_cache_dir

    level = None if level_key == "site" else SimpleNamespace(id=str(level_key))
    return clean_plan_cache_dir(project_root, level) / "plan_model.json"  # type: ignore[arg-type]


def ifc_status(project_root: Path | str, level_key: str) -> IfcStatus:
    """'missing' / 'fresh' / 'stale': the single freshness rule of jobs.ifc_export_status (manifest + plan_model sha256)."""
    from samcloud.bim import jobs as bim_jobs

    bim_dir(project_root, level_key)  # validates the level key
    return bim_jobs.ifc_export_status(project_root, level_key)


CATEGORY_ORDER: tuple[str, ...] = (
    "walls", "ceilings", "floors", "doors", "windows", "furniture", "sanitary", "appliances", "builtins", "spaces", "openings", "other",
)
HIDDEN_BY_DEFAULT: tuple[str, ...] = ("spaces", "openings", "ceilings")

_CLASS_CATEGORY: dict[str, str] = {
    "IfcWall": "walls", "IfcWallStandardCase": "walls", "IfcDoor": "doors", "IfcWindow": "windows",
    "IfcSpace": "spaces", "IfcOpeningElement": "openings", "IfcFurniture": "furniture",
    "IfcSanitaryTerminal": "sanitary", "IfcElectricAppliance": "appliances", "IfcSystemFurnitureElement": "builtins",
}
_CEILING_TYPES = {"ROOF", "USERDEFINED", "CEILING"}


def category_of(ifc_class: str, predefined_type: str | None = None) -> str:
    """Viewer category key of an IFC element (several classes/types share one checkbox)."""
    kind = (predefined_type or "").upper()
    if ifc_class == "IfcSlab":
        return "ceilings" if kind in _CEILING_TYPES else "floors"
    if ifc_class == "IfcCovering":
        return "ceilings" if kind == "CEILING" else "floors" if kind == "FLOORING" else "other"
    return _CLASS_CATEGORY.get(ifc_class, "other")


def category_index(data: dict[str, Any]) -> tuple[dict[str, str], dict[str, int]]:
    """(GlobalId -> category, category -> element count) from a parsed props.json."""
    by_node: dict[str, str] = {}
    counts: dict[str, int] = {}
    elements = data.get("elements") if isinstance(data, dict) else None
    if isinstance(elements, dict):
        for guid, info in elements.items():
            if not isinstance(info, dict):
                continue
            category = category_of(str(info.get("ifc_class") or ""), info.get("predefined_type"))
            by_node[str(guid)] = category
            counts[category] = counts.get(category, 0) + 1
    return by_node, {key: counts[key] for key in CATEGORY_ORDER if key in counts}


# --------------------------------------------------------------------------------------
# property panel
# --------------------------------------------------------------------------------------

def is_default_pset(name: str) -> bool:
    """Property sets shown without "show all": Pset_*Common / Qto_* (dimensions) and SAMcloud_Scan.

    Internal sets (``Raumlot_*``, ``Lining``, ``Panel``, ``SAMcloud_Frame`` ...) stay behind the toggle.
    """
    return name == "SAMcloud_Scan" or name.startswith("Qto_") or (name.startswith("Pset_") and name.endswith("Common"))


def visible_psets(info: dict[str, Any], show_all: bool = False) -> list[tuple[str, Any]]:
    """(name, values) property sets of one props.json element in file order; filtered unless ``show_all``."""
    psets = info.get("psets") if isinstance(info, dict) else None
    if not isinstance(psets, dict):
        return []
    return [(str(name), values) for name, values in psets.items() if show_all or is_default_pset(str(name))]


def opening_details(info: dict[str, Any]) -> dict[str, Any] | None:
    """Window / door details of one props.json element, or None for every other class.

    Keys (only those that can be determined): ``width``, ``height``, ``sill`` (BRH, parapet height above the
    storey floor), ``head`` (STH, lintel height above the storey floor), ``mullions`` (windows) and
    ``glazed`` (doors; windows only when true) and, for doors with a measured open leaf, ``hinge`` (left / right along the opening axis), ``swing`` (+normal / -normal) and ``leaf_angle`` (degrees).  Sill and head come from the element bounding box relative to the storey elevation.
    """
    if not isinstance(info, dict) or info.get("ifc_class") not in ("IfcWindow", "IfcDoor"):
        return None
    psets = info.get("psets") if isinstance(info.get("psets"), dict) else {}
    window = info.get("ifc_class") == "IfcWindow"
    quantities = psets.get("Qto_WindowBaseQuantities" if window else "Qto_DoorBaseQuantities") or {}
    location = psets.get("Raumlot_Location") or {}
    user = psets.get("Raumlot_User") or {}
    details: dict[str, Any] = {}
    for key, source in (("width", "Width"), ("height", "Height")):
        if isinstance(quantities.get(source), (int, float)):
            details[key] = float(quantities[source])
    floor, low, high = location.get("StoreyElevation"), location.get("BBoxMinZ"), location.get("BBoxMaxZ")
    if all(isinstance(value, (int, float)) for value in (floor, low, high)):
        details["sill"] = max(0.0, float(low) - float(floor))
        details["head"] = float(high) - float(floor)
    if window:
        if isinstance(user.get("MullionCount"), (int, float)):
            details["mullions"] = int(round(float(user["MullionCount"])))
        else:
            panels = sum(1 for name in psets if str(name).startswith("Panel"))
            if panels:
                details["mullions"] = max(panels - 1, 0)
    if "Glazed" in user and (not window or user["Glazed"]):
        details["glazed"] = bool(user["Glazed"])
    if not window:
        if user.get("HingeSide") in ("left", "right"):
            details["hinge"] = str(user["HingeSide"])
        if user.get("SwingNormalSide") in ("+normal", "-normal"):
            details["swing"] = str(user["SwingNormalSide"])
        if isinstance(user.get("LeafOpeningAngleDeg"), (int, float)):
            details["leaf_angle"] = float(user["LeafOpeningAngleDeg"])
    return details
