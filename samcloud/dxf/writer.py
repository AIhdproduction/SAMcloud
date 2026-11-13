"""In-memory DXF construction and audited atomic file export."""

from __future__ import annotations

from datetime import datetime
import os
from pathlib import Path
import tempfile

import ezdxf
from ezdxf.enums import TextEntityAlignment
import numpy as np

from samcloud.dxf.las_source import Frame
from samcloud.dxf.result import DxfPlanResult


def frame_scale_and_yaw(frame: Frame, coordinates: str) -> tuple[float, float]:
    """Compute length scale divisor and angle yaw for DXF entity transforms.

    When coordinates == "original":
    - Frame.to_original(p) = (p @ R) / scale + origin.
    - Every length is divided by frame.scale_factor.
    - Every angle increases by the yaw of the local->original rotation,
      computed from the transformed local x-axis direction (R.T @ [1, 0, 0])
      projected to XY, as degrees: atan2(dy, dx).
    When coordinates == "local":
    - scale divisor is 1.0;
    - yaw is 0.0 degrees.
    """
    if coordinates == "original":
        r = np.asarray(frame.rotation, dtype=np.float64)
        x_axis = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        transformed_x = r.T @ x_axis
        yaw = float(np.degrees(np.arctan2(transformed_x[1], transformed_x[0])))
        return float(frame.scale_factor), yaw
    return 1.0, 0.0


def build_dxf_document(result: DxfPlanResult, *, coordinates: str = "local") -> ezdxf.document.Drawing:
    """Build an R2018 metres DXF entirely from neutral cached entities."""
    if coordinates not in {"local", "original"}:
        raise ValueError("coordinates must be 'local' or 'original'")
    document = ezdxf.new("R2018")
    document.header["$INSUNITS"] = 6
    all_layers = dict.fromkeys(list(result.layer_colors) + list(result.layer_styles))
    for name in all_layers:
        color = result.layer_colors.get(name, 7)
        linetype = result.layer_linetypes.get(name, "CONTINUOUS")
        if linetype not in document.linetypes:
            document.linetypes.add(linetype, pattern=[0.2, -0.1], description=f"{linetype} export pattern")
        if name not in document.layers:
            layer_entity = document.layers.add(name, color=color, linetype=linetype)
        else:
            layer_entity = document.layers.get(name)
        style = result.layer_styles.get(name)
        if style:
            true_color = style.get("true_color")
            if true_color is not None:
                layer_entity.rgb = (int(true_color[0]), int(true_color[1]), int(true_color[2]))
            lineweight = style.get("lineweight")
            if lineweight is not None:
                layer_entity.dxf.lineweight = int(lineweight)
    if "X-NOTE" not in document.layers:
        document.layers.add("X-NOTE", color=8, linetype="CONTINUOUS")
    _ensure_furniture_block(document)
    if result.blocks:
        from samcloud.ortho.vector.symbols import add_symbol_blocks
        add_symbol_blocks(document, result.blocks)
    scale_div, yaw = frame_scale_and_yaw(result.frame, coordinates)
    modelspace = document.modelspace()
    for layer, entities in result.layers.items():
        for entity in entities:
            _add_entity(modelspace, entity, layer, result, coordinates, scale_div, yaw)
    lower_left = _note_point(result, coordinates)
    note_text = f"{result.project_name} | {result.level_name} | {result.profile_name or 'unknown profile'} | scale {result.scale_factor:.6g} | {datetime.now().date().isoformat()}"
    if result.variant != "pointcloud":
        note_text += f" | variant {result.variant}"
    modelspace.add_text(note_text, dxfattribs={"height": 0.12, "layer": "X-NOTE"}).set_placement(lower_left)
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


def _add_entity(
    modelspace: ezdxf.layouts.Modelspace,
    entity: dict,
    layer: str,
    result: DxfPlanResult,
    coordinates: str,
    scale_div: float | None = None,
    yaw: float | None = None,
) -> None:
    if scale_div is None or yaw is None:
        scale_div, yaw = frame_scale_and_yaw(result.frame, coordinates)
    entity_type = entity.get("type")
    attributes = {"layer": layer}
    if entity_type == "line":
        modelspace.add_line(_point(entity["start"], result, coordinates), _point(entity["end"], result, coordinates), dxfattribs=attributes)
    elif entity_type == "polyline":
        points = [_point(point, result, coordinates) for point in entity["points"]]
        heights = entity.get("z")
        if heights is not None and len(heights) == len(points):
            # One continuous 3D polyline per ring (terrain breakline); heights come from the DTM.
            modelspace.add_polyline3d(
                [_point([*point_xy[:2], height], result, coordinates) for point_xy, height in zip(entity["points"], heights, strict=True)],
                close=bool(entity.get("closed", False)), dxfattribs=attributes,
            )
            for hole, hole_heights in zip(entity.get("holes", []), entity.get("hole_z", []), strict=False):
                modelspace.add_polyline3d([_point([*point_xy[:2], height], result, coordinates) for point_xy, height in zip(hole, hole_heights, strict=True)], close=True, dxfattribs=attributes)
        else:
            modelspace.add_lwpolyline(points, close=bool(entity.get("closed", False)), dxfattribs=attributes)
            for hole in entity.get("holes", []):
                modelspace.add_lwpolyline([_point(point, result, coordinates) for point in hole], close=True, dxfattribs=attributes)
        if result.layer_hatches.get(layer, False) and entity.get("closed", False):
            hatch = modelspace.add_hatch(color=result.layer_colors.get(layer, 7), dxfattribs=attributes)
            style = result.layer_styles.get(layer, {})
            hatch_pattern = style.get("hatch_pattern")
            if hatch_pattern:
                scale = float(style.get("hatch_scale") or 1.0)
                hatch.set_pattern_fill(hatch_pattern, color=result.layer_colors.get(layer, 7), scale=scale)
            hatch.paths.add_polyline_path(points, is_closed=True)
            for hole in entity.get("holes", []):
                hatch.paths.add_polyline_path([_point(point, result, coordinates) for point in hole], is_closed=True)
    elif entity_type == "polyline3d":
        points = [_point(point, result, coordinates) for point in entity["points"]]
        modelspace.add_polyline3d(points, close=bool(entity.get("closed", False)), dxfattribs=attributes)
    elif entity_type == "circle":
        radius = float(entity["radius"]) / scale_div
        modelspace.add_circle(_point(entity["center"], result, coordinates), radius, dxfattribs=attributes)
    elif entity_type == "point":
        modelspace.add_point(_point(entity["point"], result, coordinates), dxfattribs=attributes)
    elif entity_type == "insert":
        rotation = float(entity.get("rotation", 0.0)) + yaw
        xscale = float(entity.get("xscale", 1.0)) / scale_div
        yscale = float(entity.get("yscale", 1.0)) / scale_div
        reference = modelspace.add_blockref(
            str(entity["name"]),
            _point(entity["point"], result, coordinates),
            dxfattribs={**attributes, "rotation": rotation, "xscale": xscale, "yscale": yscale},
        )
        reference.add_auto_attribs(entity.get("attributes", {}))
    elif entity_type == "arc":
        radius = float(entity["radius"]) / scale_div
        start_angle = float(entity["start_angle"]) + yaw
        end_angle = float(entity["end_angle"]) + yaw
        modelspace.add_arc(_point(entity["center"], result, coordinates), radius, start_angle, end_angle, dxfattribs=attributes)
    elif entity_type == "text":
        align_name = str(entity.get("align", "MIDDLE_CENTER"))
        align_enum = getattr(TextEntityAlignment, align_name, TextEntityAlignment.MIDDLE_CENTER)
        point = _point(entity["point"], result, coordinates)
        height = float(entity.get("height", 0.12)) / scale_div
        rotation = float(entity.get("rotation", 0.0)) + yaw
        text_entity = modelspace.add_text(
            str(entity.get("text", "")),
            height=height,
            rotation=rotation,
            dxfattribs=attributes,
        )
        text_entity.set_placement(point, align=align_enum)


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
