"""In-memory DXF construction and audited atomic file export."""

from __future__ import annotations

from datetime import datetime
from functools import lru_cache
import logging
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Iterable

import ezdxf
from ezdxf.enums import TextEntityAlignment
import numpy as np
import yaml

from samcloud.core.paths import CONFIG_DIRECTORY
from samcloud.dxf.las_source import Frame
from samcloud.dxf.result import DxfPlanResult

LOGGER = logging.getLogger(__name__)

TIN_LAYER = "S-TOPO-TIN"
MISSING_VALUE = "n/a"
OUTDOOR_BLOCK_ATTRIBUTES: dict[str, tuple[str, ...]] = {
    "TREE": ("HEIGHT", "CROWN_DIA", "TRUNK_DIA"),
    "SHRUB": ("HEIGHT",),
    "MANHOLE": ("Z_DECKEL", "TYP"),
}
_SYMBOL_LAYERS = {"TREE": "S-TREE", "SHRUB": "S-SHRUB", "MANHOLE": "S-MANH"}


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


def build_dxf_document(
    result: DxfPlanResult,
    *,
    coordinates: str = "local",
    local_origin: bool = False,
    origin_offset: tuple[float, float, float] | None = None,
) -> ezdxf.document.Drawing:
    """Build an R2018 metres DXF entirely from neutral cached entities.

    ``local_origin=False`` (default) writes the coordinates exactly as computed (real LV95 when
    ``coordinates="original"``). With ``local_origin=True`` every entity is translated by
    ``-offset`` (``origin_offset`` or, by default, the whole-metre lower-left corner of the
    background extent) and the offset is recorded in the header variable ``$INSBASE`` and in the
    X-NOTE text; true coordinate = written coordinate + offset.
    """
    if coordinates not in {"local", "original"}:
        raise ValueError("coordinates must be 'local' or 'original'")
    document = ezdxf.new("R2018")
    document.header["$INSUNITS"] = 6
    _ensure_used_outdoor_items(document, result)
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
    # Draw order = entity order: area fills (wall poché) first, texts last, so no fill ever covers a label.
    for stage in (0, 1, 2):
        for layer, entities in result.layers.items():
            for entity in entities:
                if _draw_stage(entity) == stage:
                    _add_entity(modelspace, entity, layer, result, coordinates, scale_div, yaw)
    lower_left = _note_point(result, coordinates)
    note_text = f"{result.project_name} | {result.level_name} | {result.profile_name or 'unknown profile'} | scale {result.scale_factor:.6g} | {datetime.now().date().isoformat()}"
    if result.variant != "pointcloud":
        note_text += f" | variant {result.variant}"
    offset: tuple[float, float, float] | None = None
    if local_origin:
        offset = _resolve_offset(origin_offset, lower_left)
        note_text += f" | local origin offset E {offset[0]:.3f} N {offset[1]:.3f} Z {offset[2]:.3f} (true = written + offset)"
    modelspace.add_text(note_text, dxfattribs={"height": 0.12, "layer": "X-NOTE"}).set_placement(lower_left)
    if offset is not None:
        document.header["$INSBASE"] = offset
        for drawn in modelspace:
            drawn.translate(-offset[0], -offset[1], -offset[2])
    _audit(document)
    return document


def _resolve_offset(
    origin_offset: tuple[float, float, float] | None, lower_left: tuple[float, float, float]
) -> tuple[float, float, float]:
    """Return the explicit offset or the lower-left corner rounded down to whole metres."""
    if origin_offset is not None:
        return (float(origin_offset[0]), float(origin_offset[1]), float(origin_offset[2]) if len(origin_offset) > 2 else 0.0)
    return (float(math.floor(lower_left[0])), float(math.floor(lower_left[1])), 0.0)


@lru_cache(maxsize=4)
def _load_extra_layers(path: str) -> tuple[dict[str, Any], ...]:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    return tuple(dict(record) for record in raw.get("outdoor_extra_layers", []))


def ensure_outdoor_layers(
    document: ezdxf.document.Drawing,
    names: Iterable[str] | None = None,
    *,
    tin_frozen: bool | None = None,
    config_path: Path | None = None,
) -> list[str]:
    """Create the drone-pipeline outdoor layers (``outdoor_extra_layers`` in dxf_layers.yaml).

    ``names=None`` creates all of them. Existing layers are left untouched. ``frozen: true``
    layers (S-TOPO-TIN) are created frozen and off; ``tin_frozen`` overrides that for S-TOPO-TIN.
    Returns the names of the layers that were newly created.
    """
    source = Path(config_path) if config_path else CONFIG_DIRECTORY / "dxf_layers.yaml"
    wanted = None if names is None else set(names)
    created: list[str] = []
    for record in _load_extra_layers(str(source)):
        name = str(record["name"])
        if (wanted is not None and name not in wanted) or name in document.layers:
            continue
        layer = document.layers.add(name, color=int(record.get("color", 7)), linetype=str(record.get("linetype", "CONTINUOUS")))
        if record.get("lineweight") is not None:
            layer.dxf.lineweight = int(record["lineweight"])
        frozen = bool(record.get("frozen", False))
        if name == TIN_LAYER and tin_frozen is not None:
            frozen = tin_frozen
        if frozen:
            layer.freeze()
            layer.off()
        created.append(name)
    return created


def _ensure_used_outdoor_items(document: ezdxf.document.Drawing, result: DxfPlanResult) -> None:
    """Create outdoor layers and symbol blocks only when the result references them (indoor output unchanged)."""
    extra_names = {str(record["name"]) for record in _load_extra_layers(str(CONFIG_DIRECTORY / "dxf_layers.yaml"))}
    used_layers = {name for name in result.layers if name in extra_names}
    used_blocks: set[str] = set()
    for entities in result.layers.values():
        for entity in entities:
            if entity.get("type") == "insert" and entity.get("name") in OUTDOOR_BLOCK_ATTRIBUTES:
                used_blocks.add(str(entity["name"]))
    if used_layers:
        ensure_outdoor_layers(document, used_layers)
    if used_blocks:
        ensure_outdoor_blocks(document, used_blocks)


def ensure_outdoor_blocks(document: ezdxf.document.Drawing, names: Iterable[str] | None = None) -> None:
    """Define the TREE, SHRUB and MANHOLE blocks once per document (unit-size symbols with ATTDEFs).

    The TREE and SHRUB circles have radius 1 and are scaled by the INSERT scale (crown radius);
    attribute definitions are invisible so they do not scale into large labels.
    """
    wanted = set(OUTDOOR_BLOCK_ATTRIBUTES) if names is None else set(names)
    if "TREE" in wanted and "TREE" not in document.blocks:
        block = document.blocks.new("TREE")
        block.add_circle((0.0, 0.0), 1.0)
        block.add_line((-0.1, 0.0), (0.1, 0.0))
        block.add_line((0.0, -0.1), (0.0, 0.1))
        _add_invisible_attdefs(block, OUTDOOR_BLOCK_ATTRIBUTES["TREE"])
    if "SHRUB" in wanted and "SHRUB" not in document.blocks:
        block = document.blocks.new("SHRUB")
        block.add_circle((0.0, 0.0), 1.0)
        _add_invisible_attdefs(block, OUTDOOR_BLOCK_ATTRIBUTES["SHRUB"])
    if "MANHOLE" in wanted and "MANHOLE" not in document.blocks:
        block = document.blocks.new("MANHOLE")
        block.add_circle((0.0, 0.0), 0.3)
        block.add_line((-0.3, 0.0), (0.3, 0.0))
        block.add_line((0.0, -0.3), (0.0, 0.3))
        _add_invisible_attdefs(block, OUTDOOR_BLOCK_ATTRIBUTES["MANHOLE"])


def _add_invisible_attdefs(block: Any, tags: Iterable[str]) -> None:
    for index, tag in enumerate(tags):
        block.add_attdef(tag, (0.0, -0.1 * (index + 1)), MISSING_VALUE, height=0.05, dxfattribs={"flags": 1})


def _attribute_text(value: object) -> str:
    if value is None:
        return MISSING_VALUE
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.2f}" if math.isfinite(float(value)) else MISSING_VALUE
    text = str(value).strip()
    return text or MISSING_VALUE


def outdoor_symbol_insert(
    kind: str,
    xyz: Iterable[float],
    attributes: dict[str, object] | None = None,
    crown_radius: float | None = None,
) -> dict[str, Any]:
    """Return the ``insert`` entity dict for a TREE, SHRUB or MANHOLE symbol.

    Every attribute of the block is written; missing values become "n/a" (never estimated).
    The block scale is the measured crown radius (TREE/SHRUB); without a radius the scale is 1.
    A TREE without an explicit CROWN_DIA gets ``2 * crown_radius`` (a measured value).
    """
    name = str(kind).upper()
    if name not in OUTDOOR_BLOCK_ATTRIBUTES:
        raise ValueError(f"unknown outdoor symbol kind '{kind}'")
    given = {str(key).upper(): value for key, value in (attributes or {}).items()}
    radius = float(crown_radius) if crown_radius is not None and math.isfinite(float(crown_radius)) and float(crown_radius) > 0 else None
    if name == "TREE" and radius is not None and given.get("CROWN_DIA") is None:
        given["CROWN_DIA"] = 2.0 * radius
    texts = {tag: _attribute_text(given.get(tag)) for tag in OUTDOOR_BLOCK_ATTRIBUTES[name]}
    scale = radius if (radius is not None and name in {"TREE", "SHRUB"}) else 1.0
    point = list(xyz)
    return {
        "type": "insert",
        "name": name,
        "layer": _SYMBOL_LAYERS[name],
        "point": [float(point[0]), float(point[1]), float(point[2]) if len(point) > 2 else 0.0],
        "xscale": scale,
        "yscale": scale,
        "rotation": 0.0,
        "attributes": texts,
    }


def contours_as_splines(entities: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert ``polyline3d`` contour entities to SPLINE fit-point entities.

    Layer and other keys are kept, Z values are kept, closed contours stay closed (a duplicated
    closing vertex is dropped). Entities of other types are passed through unchanged.
    """
    converted: list[dict[str, Any]] = []
    for entity in entities:
        if entity.get("type") != "polyline3d":
            converted.append(entity)
            continue
        points = [[float(v) for v in point] for point in entity["points"]]
        closed = bool(entity.get("closed", False))
        if closed and len(points) > 1 and math.dist(points[0], points[-1]) < 1e-9:
            points = points[:-1]
        spline = {key: value for key, value in entity.items() if key != "points"}
        spline.update({"type": "spline", "fit_points": points, "closed": closed})
        converted.append(spline)
    return converted


def assert_no_tin_outside(doc: ezdxf.document.Drawing, allowed_layer: str = TIN_LAYER) -> list[Any]:
    """Return every 3DFACE (TIN mesh) entity that is not on ``allowed_layer`` (empty list = clean)."""
    return [entity for entity in doc.modelspace() if entity.dxftype() == "3DFACE" and entity.dxf.layer != allowed_layer]


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


def _draw_stage(entity: dict) -> int:
    """0 for hatch fills, 2 for texts, 1 for all other geometry."""
    kind = entity.get("type")
    if kind == "hatch":
        return 0
    if kind == "text":
        return 2
    return 1


def _ensure_furniture_block(document: ezdxf.document.Drawing) -> None:
    if "FURNITURE" in document.blocks:
        return
    block = document.blocks.new("FURNITURE")
    block.add_lwpolyline([(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)], close=True)
    block.add_attdef("class", (0.0, -0.15), "", height=0.08)
    block.add_attdef("height", (0.0, -0.3), "", height=0.08)
    block.add_attdef("instance", (0.0, -0.45), "", height=0.08)
    block.add_attdef("label", (0.0, -0.6), "", height=0.08)


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
    elif entity_type == "hatch":
        # Area fill only (for example wall poché); the outline is a separate polyline entity.
        points = [_point(point, result, coordinates) for point in entity.get("points", [])]
        if len(points) >= 3:
            style = result.layer_styles.get(layer, {})
            hatch = modelspace.add_hatch(color=256, dxfattribs=attributes)
            hatch_pattern = style.get("hatch_pattern")
            if hatch_pattern and str(hatch_pattern).upper() != "SOLID":
                hatch.set_pattern_fill(str(hatch_pattern), color=256, scale=float(style.get("hatch_scale") or 1.0))
            hatch.paths.add_polyline_path([point[:2] for point in points], is_closed=True)
            for hole in entity.get("holes", []):
                if len(hole) >= 3:
                    hatch.paths.add_polyline_path([_point(point, result, coordinates)[:2] for point in hole], is_closed=True)
    elif entity_type == "polyline3d":
        points = [_point(point, result, coordinates) for point in entity["points"]]
        modelspace.add_polyline3d(points, close=bool(entity.get("closed", False)), dxfattribs=attributes)
    elif entity_type == "lwpolyline":
        # Elevated 2D polyline: x/y from the points, constant elevation from `elevation` (or the first point z).
        raw = [list(point) for point in entity["points"]]
        elevation = entity.get("elevation")
        points3 = [
            _point([p[0], p[1], float(elevation) if elevation is not None else (p[2] if len(p) > 2 else 0.0)], result, coordinates)
            for p in raw
        ]
        z_value = points3[0][2] if points3 else 0.0
        modelspace.add_lwpolyline([point[:2] for point in points3], close=bool(entity.get("closed", False)), dxfattribs={**attributes, "elevation": z_value})
        if entity.get("hatch", False) and entity.get("closed", False) and len(points3) >= 3:
            # Optional boundary hatch, off unless the entity asks for it.
            hatch = modelspace.add_hatch(color=256, dxfattribs={**attributes, "elevation": (0.0, 0.0, z_value)})
            hatch.paths.add_polyline_path([point[:2] for point in points3], is_closed=True)
    elif entity_type == "3dface":
        corners = [_point(point, result, coordinates) for point in entity["points"]]
        if len(corners) == 3:
            corners.append(corners[2])
        modelspace.add_3dface(corners, dxfattribs=attributes)
    elif entity_type == "spline":
        fit_points = [_point(point, result, coordinates) for point in entity["fit_points"]]
        spline = modelspace.add_spline(fit_points=fit_points, dxfattribs=attributes)
        if entity.get("closed", False):
            spline.closed = True
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
