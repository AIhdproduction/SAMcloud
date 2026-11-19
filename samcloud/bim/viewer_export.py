"""IFC -> binary glTF 2.0 (GLB) plus a properties JSON for the browser IFC viewer.

Contract (consumed by the viewer UI):

* ``model.glb``: one node per IfcProduct that has geometry, ``node.name`` = GlobalId,
  ``node.extras`` = ``{"ifc_class", "name"}``. One mesh per node (POSITION + NORMAL, uint32
  indices); one primitive per IFC surface style. World coordinates in metres, converted from
  the IFC Z-up frame to glTF Y-up: ``(x, y, z) -> (x, z, -y)``.
* ``props.json``: ``schema_version``, ``ifc_file``, ``classes`` (node count per class) and
  ``elements`` (per GlobalId: class, name, predefined/object type, storey, property sets and
  quantities).

The GLB is written directly with numpy + struct (no glTF dependency).
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
import json
import math
import os
from pathlib import Path
import struct
from typing import Any

import numpy as np

PROPS_SCHEMA_VERSION = 1
GLB_NAME = "model.glb"
PROPS_NAME = "props.json"

# Fallback colours (RGBA) when ifcopenshell reports no usable style colour.
CLASS_COLOURS: dict[str, tuple[float, float, float, float]] = {
    "IfcWall": (0.90, 0.90, 0.88, 1.0),
    "IfcWallStandardCase": (0.90, 0.90, 0.88, 1.0),
    "IfcSlab": (0.55, 0.55, 0.55, 1.0),
    "IfcDoor": (0.55, 0.30, 0.15, 1.0),
    "IfcWindow": (0.65, 0.80, 0.90, 0.35),
    "IfcSpace": (0.65, 0.75, 0.80, 0.20),
    "IfcOpeningElement": (0.90, 0.30, 0.30, 0.30),
    "IfcFurnishingElement": (0.70, 0.55, 0.40, 1.0),
    "IfcFurniture": (0.70, 0.55, 0.40, 1.0),
}
DEFAULT_COLOUR = (0.70, 0.70, 0.70, 1.0)
TRANSLUCENT_CLASSES = {"IfcSpace", "IfcOpeningElement"}


@dataclass(frozen=True)
class ViewerAssets:
    glb_path: Path
    props_path: Path
    counts: dict[str, int] = field(default_factory=dict)


# --------------------------------------------------------------------------
# geometry extraction
# --------------------------------------------------------------------------

def _rgba_for(material: Any, ifc_class: str) -> tuple[float, float, float, float]:
    base = CLASS_COLOURS.get(ifc_class, DEFAULT_COLOUR)
    rgb = base[:3]
    alpha = base[3]
    try:
        diffuse = material.diffuse
        candidate = (float(diffuse.r()), float(diffuse.g()), float(diffuse.b()))
        if all(math.isfinite(value) for value in candidate):
            rgb = candidate
    except Exception:
        pass
    try:
        if material.has_transparency():
            transparency = float(material.transparency)
            if math.isfinite(transparency):
                alpha = 1.0 - max(0.0, min(1.0, transparency))
    except Exception:
        pass
    if ifc_class in TRANSLUCENT_CLASSES:
        alpha = min(alpha, CLASS_COLOURS[ifc_class][3])
    return (round(rgb[0], 4), round(rgb[1], 4), round(rgb[2], 4), round(alpha, 4))


def _flat_primitives(shape: Any, ifc_class: str) -> list[tuple[np.ndarray, np.ndarray, tuple[float, float, float, float]]]:
    """Return (positions Nx3 float64 Y-up, normals Nx3 float32 Y-up, rgba) per style; faces are unwelded."""
    geometry = shape.geometry
    verts = np.asarray(geometry.verts, dtype=np.float64).reshape(-1, 3)
    faces = np.asarray(geometry.faces, dtype=np.int64).reshape(-1, 3)
    if len(verts) == 0 or len(faces) == 0:
        return []
    material_ids = np.asarray(geometry.material_ids, dtype=np.int64)
    if len(material_ids) != len(faces):
        material_ids = np.zeros(len(faces), dtype=np.int64)
    materials = list(geometry.materials)
    result: list[tuple[np.ndarray, np.ndarray, tuple[float, float, float, float]]] = []
    for material_index in np.unique(material_ids):
        selected = faces[material_ids == material_index]
        triangles = verts[selected]  # (F, 3, 3) in IFC Z-up
        yup = np.stack([triangles[..., 0], triangles[..., 2], -triangles[..., 1]], axis=-1)
        normal = np.cross(yup[:, 1] - yup[:, 0], yup[:, 2] - yup[:, 0])
        length = np.linalg.norm(normal, axis=1, keepdims=True)
        keep = length[:, 0] > 1e-15
        if not keep.any():
            continue
        yup, normal, length = yup[keep], normal[keep], length[keep]
        normal = normal / length
        positions = yup.reshape(-1, 3)
        normals = np.repeat(normal, 3, axis=0).astype(np.float32)
        if 0 <= int(material_index) < len(materials):
            rgba = _rgba_for(materials[int(material_index)], ifc_class)
        else:
            rgba = _rgba_for(None, ifc_class)
        result.append((positions, normals, rgba))
    return result


def _iter_shapes(model: Any, products: list[Any], threads: int):
    import ifcopenshell.geom

    settings = ifcopenshell.geom.settings()
    settings.set("use-world-coords", True)
    iterator = ifcopenshell.geom.iterator(settings, model, threads, include=products)
    if not iterator.initialize():
        return
    while True:
        yield iterator.get()
        if not iterator.next():
            break


# --------------------------------------------------------------------------
# GLB writer
# --------------------------------------------------------------------------

def _pad(data: bytes, filler: bytes) -> bytes:
    remainder = len(data) % 4
    return data if remainder == 0 else data + filler * (4 - remainder)


def _build_glb(
    nodes: list[tuple[str, dict[str, Any], list[tuple[np.ndarray, np.ndarray, tuple[float, float, float, float]]]]],
    offset: np.ndarray | None = None,
) -> bytes:
    binary = bytearray()
    buffer_views: list[dict[str, Any]] = []
    accessors: list[dict[str, Any]] = []
    meshes: list[dict[str, Any]] = []
    gltf_nodes: list[dict[str, Any]] = []
    materials: list[dict[str, Any]] = []
    material_index: dict[tuple[float, float, float, float], int] = {}

    def add_view(array: np.ndarray, target: int) -> int:
        data = array.tobytes()
        offset = len(binary)
        binary.extend(data)
        binary.extend(b"\x00" * ((-len(binary)) % 4))
        buffer_views.append({"buffer": 0, "byteOffset": offset, "byteLength": len(data), "target": target})
        return len(buffer_views) - 1

    def material_for(rgba: tuple[float, float, float, float], label: str) -> int:
        key = rgba
        if key not in material_index:
            material: dict[str, Any] = {
                "name": f"{label}_{len(materials)}",
                "pbrMetallicRoughness": {"baseColorFactor": list(rgba), "metallicFactor": 0.0, "roughnessFactor": 0.9},
                "doubleSided": True,
            }
            if rgba[3] < 0.999:
                material["alphaMode"] = "BLEND"
            material_index[key] = len(materials)
            materials.append(material)
        return material_index[key]

    for guid, extras, primitives in nodes:
        gltf_primitives: list[dict[str, Any]] = []
        for positions, normals, rgba in primitives:
            if offset is not None:
                positions = positions - offset
            positions = positions.astype(np.float32)
            count = len(positions)
            indices = np.arange(count, dtype=np.uint32)
            position_view = add_view(positions, 34962)
            normal_view = add_view(normals, 34962)
            index_view = add_view(indices, 34963)
            accessors.append({
                "bufferView": position_view, "componentType": 5126, "count": count, "type": "VEC3",
                "min": [float(value) for value in positions.min(axis=0)],
                "max": [float(value) for value in positions.max(axis=0)],
            })
            position_accessor = len(accessors) - 1
            accessors.append({"bufferView": normal_view, "componentType": 5126, "count": count, "type": "VEC3"})
            normal_accessor = len(accessors) - 1
            accessors.append({"bufferView": index_view, "componentType": 5125, "count": count, "type": "SCALAR"})
            index_accessor = len(accessors) - 1
            gltf_primitives.append({
                "attributes": {"POSITION": position_accessor, "NORMAL": normal_accessor},
                "indices": index_accessor,
                "material": material_for(rgba, str(extras.get("ifc_class", "Material"))),
                "mode": 4,
            })
        meshes.append({"name": guid, "primitives": gltf_primitives})
        gltf_nodes.append({"name": guid, "mesh": len(meshes) - 1, "extras": extras})

    document: dict[str, Any] = {
        "asset": {"version": "2.0", "generator": "SAMcloud viewer_export"},
        "scene": 0,
        "scenes": [{"name": "model", "nodes": list(range(len(gltf_nodes)))}],
        "nodes": gltf_nodes,
        "meshes": meshes,
        "materials": materials,
        "accessors": accessors,
        "bufferViews": buffer_views,
        "buffers": [{"byteLength": len(binary)}],
    }
    for key in ("materials", "accessors", "bufferViews", "meshes", "nodes"):
        if not document[key]:
            del document[key]
    if offset is not None:
        document["asset"]["extras"] = {"origin_offset_yup": [float(value) for value in offset]}
    if not binary:
        document["buffers"] = []
    json_chunk = _pad(json.dumps(document, separators=(",", ":"), ensure_ascii=True).encode("ascii"), b" ")
    bin_chunk = _pad(bytes(binary), b"\x00")
    chunks = struct.pack("<II", len(json_chunk), 0x4E4F534A) + json_chunk
    if bin_chunk:
        chunks += struct.pack("<II", len(bin_chunk), 0x004E4942) + bin_chunk
    return struct.pack("<III", 0x46546C67, 2, 12 + len(chunks)) + chunks


# --------------------------------------------------------------------------
# properties
# --------------------------------------------------------------------------

def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (bool, str, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return _json_safe(float(value))
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "is_a") and hasattr(value, "id"):
        try:
            return {"ifc_class": value.is_a(), "id": int(value.id())}
        except Exception:
            return str(value)
    return str(value)


def _storey_name(element: Any) -> str | None:
    import ifcopenshell.util.element as util

    current = element
    for _ in range(6):
        if current is None:
            return None
        if current.is_a("IfcBuildingStorey"):
            return str(current.Name) if current.Name else str(current.GlobalId)
        container = util.get_container(current)
        if container is not None:
            current = container
            continue
        aggregate = util.get_aggregate(current)
        if aggregate is None and current.is_a("IfcOpeningElement"):
            voids = getattr(current, "VoidsElements", None)
            aggregate = voids[0].RelatingBuildingElement if voids else None
        current = aggregate
    return None


def _element_properties(element: Any) -> dict[str, Any]:
    import ifcopenshell.util.element as util

    try:
        psets = util.get_psets(element)
    except Exception:
        psets = {}
    cleaned: dict[str, dict[str, Any]] = {}
    for name, values in psets.items():
        cleaned[str(name)] = {str(key): _json_safe(item) for key, item in values.items() if key != "id"}
    return {
        "ifc_class": element.is_a(),
        "name": getattr(element, "Name", None),
        "predefined_type": _json_safe(getattr(element, "PredefinedType", None)),
        "object_type": _json_safe(getattr(element, "ObjectType", None)),
        "storey": _storey_name(element),
        "psets": cleaned,
    }


# --------------------------------------------------------------------------
# public API
# --------------------------------------------------------------------------

def _atomic_write(path: Path, data: bytes) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_bytes(data)
    os.replace(temporary, path)


def export_viewer_assets(ifc_path: Path, out_dir: Path, *, threads: int | None = None) -> ViewerAssets:
    """Write ``model.glb`` and ``props.json`` for ``ifc_path`` into ``out_dir``."""
    import ifcopenshell

    ifc_path = Path(ifc_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model = ifcopenshell.open(str(ifc_path))
    products = [element for element in model.by_type("IfcProduct") if element.Representation is not None]
    worker_threads = threads if threads is not None else max(1, min(4, (os.cpu_count() or 2) - 1))

    nodes: list[tuple[str, dict[str, Any], list[tuple[np.ndarray, np.ndarray, tuple[float, float, float, float]]]]] = []
    elements: dict[str, dict[str, Any]] = {}
    counts: Counter[str] = Counter()
    seen: set[str] = set()
    for shape in _iter_shapes(model, products, worker_threads):
        element = model.by_id(shape.id)
        guid = str(element.GlobalId)
        if guid in seen:
            continue
        ifc_class = element.is_a()
        primitives = _flat_primitives(shape, ifc_class)
        if not primitives:
            continue
        seen.add(guid)
        extras = {"ifc_class": ifc_class, "name": element.Name or ""}
        nodes.append((guid, extras, primitives))
        elements[guid] = _element_properties(element)
        counts[ifc_class] += 1

    # float32 vertices lose centimetres at survey-grid magnitudes: shift such models to a local origin.
    offset: np.ndarray | None = None
    if nodes:
        low = np.min([primitive[0].min(axis=0) for _g, _e, prims in nodes for primitive in prims], axis=0)
        high = np.max([primitive[0].max(axis=0) for _g, _e, prims in nodes for primitive in prims], axis=0)
        if float(np.max(np.abs(np.concatenate([low, high])))) > 10000.0:
            offset = np.floor((low + high) / 2.0)

    glb_path = out_dir / GLB_NAME
    props_path = out_dir / PROPS_NAME
    _atomic_write(glb_path, _build_glb(nodes, offset))
    props = {
        "schema_version": PROPS_SCHEMA_VERSION,
        "ifc_file": ifc_path.name,
        "origin_offset_yup": [float(value) for value in offset] if offset is not None else [0.0, 0.0, 0.0],
        "classes": dict(sorted(counts.items())),
        "elements": elements,
    }
    _atomic_write(props_path, json.dumps(props, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))
    return ViewerAssets(glb_path=glb_path, props_path=props_path, counts=dict(sorted(counts.items())))
