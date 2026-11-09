"""In-memory DXF construction and audited atomic file export."""

from __future__ import annotations

from datetime import datetime
import os
from pathlib import Path
import tempfile

import ezdxf
import numpy as np

from samcloud.dxf.result import DxfPlanResult


def build_dxf_document(result: DxfPlanResult, *, coordinates: str = "local") -> ezdxf.document.Drawing:
    """Build an R2018 metres DXF entirely from neutral cached entities."""
    if coordinates not in {"local", "original"}:
        raise ValueError("coordinates must be 'local' or 'original'")
    document = ezdxf.new("R2018")
    document.header["$INSUNITS"] = 6
    for name, color in result.layer_colors.items():
        linetype = result.layer_linetypes.get(name, "CONTINUOUS")
        if linetype not in document.linetypes:
            document.linetypes.add(linetype, pattern=[0.2, -0.1], description=f"{linetype} export pattern")
        if name not in document.layers:
            document.layers.add(name, color=color, linetype=linetype)
    if "X-NOTE" not in document.layers:
        document.layers.add("X-NOTE", color=8, linetype="CONTINUOUS")
    _ensure_furniture_block(document)
    modelspace = document.modelspace()
    for layer, entities in result.layers.items():
        for entity in entities:
            _add_entity(modelspace, entity, layer, result, coordinates)
    lower_left = _note_point(result, coordinates)
    modelspace.add_text(f"{result.project_name} | {result.level_name} | {result.profile_name or 'unknown profile'} | scale {result.scale_factor:.6g} | {datetime.now().date().isoformat()}", dxfattribs={"height": 0.12, "layer": "X-NOTE"}).set_placement(lower_left)
    _audit(document)
    return document


def sanitize_dxf_name(name: str) -> str:
    """Sanitize a name for Windows filenames by replacing invalid and control chars."""
    invalid = set('<>:"/\\|?*')
    cleaned = [("_" if char in invalid or ord(char) < 32 else char) for char in name]
    result = "".join(cleaned).rstrip(". ")
    return result or "_"


def export_plan(result: DxfPlanResult, out_dir: Path, coordinates: str = "local") -> Path:
    """Atomically export exactly the reviewed result and verify the saved DXF."""
    destination_directory = Path(out_dir)
    destination_directory.mkdir(parents=True, exist_ok=True)
    suffix = "site" if result.level_name == "site" else sanitize_dxf_name(result.level_name)
    safe_project = sanitize_dxf_name(result.project_name)
    destination = destination_directory / f"{safe_project}_{suffix}.dxf"
    document = build_dxf_document(result, coordinates=coordinates)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{destination.stem}.", suffix=".tmp", dir=destination_directory)
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        document.saveas(temporary)
        _audit(ezdxf.readfile(temporary))
        os.replace(temporary, destination)
    except Exception as error:
        if temporary.exists():
            temporary.unlink()
        raise RuntimeError(f"DXF export audit failed: {error}") from error
    return destination


def _ensure_furniture_block(document: ezdxf.document.Drawing) -> None:
    if "FURNITURE" in document.blocks:
        return
    block = document.blocks.new("FURNITURE")
    block.add_lwpolyline([(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)], close=True)
    block.add_attdef("class", (0.0, -0.15), "", height=0.08)
    block.add_attdef("height", (0.0, -0.3), "", height=0.08)


def _add_entity(modelspace: ezdxf.layouts.Modelspace, entity: dict, layer: str, result: DxfPlanResult, coordinates: str) -> None:
    entity_type = entity.get("type")
    attributes = {"layer": layer}
    if entity_type == "line":
        modelspace.add_line(_point(entity["start"], result, coordinates), _point(entity["end"], result, coordinates), dxfattribs=attributes)
    elif entity_type == "polyline":
        points = [_point(point, result, coordinates) for point in entity["points"]]
        modelspace.add_lwpolyline(points, close=bool(entity.get("closed", False)), dxfattribs=attributes)
        for hole in entity.get("holes", []):
            modelspace.add_lwpolyline([_point(point, result, coordinates) for point in hole], close=True, dxfattribs=attributes)
        if result.layer_hatches.get(layer, False) and entity.get("closed", False):
            hatch = modelspace.add_hatch(color=result.layer_colors[layer], dxfattribs=attributes)
            hatch.paths.add_polyline_path(points, is_closed=True)
            for hole in entity.get("holes", []):
                hatch.paths.add_polyline_path([_point(point, result, coordinates) for point in hole], is_closed=True)
    elif entity_type == "circle":
        modelspace.add_circle(_point(entity["center"], result, coordinates), float(entity["radius"]), dxfattribs=attributes)
    elif entity_type == "point":
        modelspace.add_point(_point(entity["point"], result, coordinates), dxfattribs=attributes)
    elif entity_type == "insert":
        reference = modelspace.add_blockref(str(entity["name"]), _point(entity["point"], result, coordinates), dxfattribs={**attributes, "rotation": float(entity.get("rotation", 0.0)), "xscale": float(entity.get("xscale", 1.0)), "yscale": float(entity.get("yscale", 1.0))})
        reference.add_auto_attribs(entity.get("attributes", {}))
    elif entity_type == "arc":
        modelspace.add_arc(_point(entity["center"], result, coordinates), float(entity["radius"]), float(entity["start_angle"]), float(entity["end_angle"]), dxfattribs=attributes)


def _point(value: object, result: DxfPlanResult, coordinates: str) -> tuple[float, float, float]:
    raw = list(value)
    local = (float(raw[0]), float(raw[1]), float(raw[2]) if len(raw) > 2 else 0.0)
    if coordinates == "original":
        original = result.frame.to_original(np.asarray([local], dtype=np.float64))[0]
        return (float(original[0]), float(original[1]), float(original[2]))
    return local


def _note_point(result: DxfPlanResult, coordinates: str) -> tuple[float, float, float]:
    left, bottom, _right, _top = result.background.bounds
    return _point((left, bottom, 0.0), result, coordinates)


def _audit(document: ezdxf.document.Drawing) -> None:
    auditor = document.audit()
    if auditor.has_errors:
        details = "; ".join(error.message for error in auditor.errors)
        raise RuntimeError(details or "DXF auditor reported errors")
