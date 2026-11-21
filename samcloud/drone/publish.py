"""Publish the outdoor (drone) exports into the layout the SAMcloud app already reads for level key "site".

``products/environment/`` holds ``outdoor.dxf``, ``outdoor.ifc``, ``environment.glb`` and ``classified.laz``. The DXF page
looks for ``products/dxf/export/*Site*clean.dxf`` and ``products/bim/site/`` (IFC, ``model.glb``, ``props.json``,
``export.json``). :func:`publish_environment_exports` COPIES the exports there (sources stay untouched, idempotent).

CLI: ``python -m samcloud.drone.publish <project_dir>``.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import logging
import os
from pathlib import Path
import math
import shutil
import struct
import sys
from typing import Any, Mapping

LOGGER = logging.getLogger(__name__)

SITE_KEY = "site"
SOURCE_NAME = "environment"
COORDINATES = "LV95"
_SOURCE_FILES: dict[str, str] = {
    "ifc": "outdoor.ifc",
    "dxf": "outdoor.dxf",
    "glb": "environment.glb",
    "laz": "classified.laz",
}


def environment_exports(project_dir: str | Path) -> dict[str, Path]:
    """The existing export files below ``products/environment/`` (keys ifc, dxf, glb, laz)."""
    folder = Path(project_dir) / "products" / "environment"
    found = {key: folder / name for key, name in _SOURCE_FILES.items()}
    return {key: path for key, path in found.items() if path.is_file()}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _copy(source: Path, target: Path) -> Path:
    """Copy ``source`` to ``target`` atomically (temporary file in the target folder, then replace)."""
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.tmp")
    shutil.copyfile(source, temporary)
    os.replace(temporary, target)
    return target


def _write_json(path: Path, data: Any, **dump_kwargs: Any) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(data, **dump_kwargs), encoding="utf-8")
    os.replace(temporary, path)


def _ifc_props(ifc_path: Path, ifc_name: str) -> dict[str, Any]:
    """props.json content in the shape of ``bim.viewer_export`` (GlobalId -> class, name, psets)."""
    import ifcopenshell

    from samcloud.bim import viewer_export

    model = ifcopenshell.open(str(ifc_path))
    elements: dict[str, dict[str, Any]] = {}
    counts: Counter[str] = Counter()
    for element in model.by_type("IfcProduct"):
        if element.Representation is None:
            continue
        guid = str(element.GlobalId)
        if guid in elements:
            continue
        elements[guid] = viewer_export._element_properties(element)
        counts[element.is_a()] += 1
    return {
        "schema_version": viewer_export.PROPS_SCHEMA_VERSION,
        "ifc_file": ifc_name,
        "origin_offset_yup": [0.0, 0.0, 0.0],
        "classes": dict(sorted(counts.items())),
        "elements": elements,
        "source": SOURCE_NAME,
    }


def _strict_loads_glb(data: bytes) -> tuple[dict[str, Any], bytes]:
    """(JSON document, binary chunk) of a GLB; raises ValueError for NaN/Infinity in the JSON or a broken container."""
    if len(data) < 20 or struct.unpack_from("<I", data, 0)[0] != 0x46546C67:
        raise ValueError("not a GLB file")
    length, kind = struct.unpack_from("<II", data, 12)
    if kind != 0x4E4F534A:
        raise ValueError("first GLB chunk is not JSON")

    def reject(token: str) -> Any:
        raise ValueError(f"non-finite number {token} in GLB JSON")

    document = json.loads(data[20:20 + length].decode("utf-8"), parse_constant=reject)
    binary = b""
    offset = 20 + length
    if offset + 8 <= len(data):
        bin_length, bin_kind = struct.unpack_from("<II", data, offset)
        if bin_kind == 0x004E4942:
            binary = data[offset + 8:offset + 8 + bin_length]
    return document, binary


def _pad4(data: bytes, filler: bytes) -> bytes:
    return data + filler * ((-len(data)) % 4)


def sanitize_glb(data: bytes) -> bytes:
    """GLB bytes without meshes that have non-finite vertices (or NaN accessor bounds); strict JSON, no NaN."""
    import numpy as np

    raw_length, _kind = struct.unpack_from("<II", data, 12)
    document = json.loads(data[20:20 + raw_length].decode("utf-8"))  # tolerant: Python accepts NaN
    binary = b""
    if 20 + raw_length + 8 <= len(data):
        bin_length, bin_kind = struct.unpack_from("<II", data, 20 + raw_length)
        if bin_kind == 0x004E4942:
            binary = data[28 + raw_length:28 + raw_length + bin_length]
    accessors = document.get("accessors", [])
    views = document.get("bufferViews", [])

    def bounds_bad(accessor: dict[str, Any]) -> bool:
        return any(
            not isinstance(v, (int, float)) or not math.isfinite(v)
            for key in ("min", "max") for v in accessor.get(key, [])
        )

    def accessor_bad(index: int) -> bool:
        accessor = accessors[index]
        if bounds_bad(accessor):
            return True
        if accessor.get("componentType") == 5126 and "bufferView" in accessor:
            view = views[accessor["bufferView"]]
            width = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4}.get(accessor.get("type"), 1)
            start = int(view.get("byteOffset", 0)) + int(accessor.get("byteOffset", 0))
            count = int(accessor["count"]) * width
            if start + 4 * count > len(binary):
                return True
            values = np.frombuffer(binary, dtype="<f4", count=count, offset=start)
            return not bool(np.isfinite(values).all())
        return False

    bad_meshes: set[int] = set()
    for index, mesh in enumerate(document.get("meshes", [])):
        for primitive in mesh.get("primitives", []):
            position = primitive.get("attributes", {}).get("POSITION")
            if position is not None and accessor_bad(position):
                bad_meshes.add(index)
    for accessor in accessors:  # leftover bounds of dropped meshes must still be valid JSON
        if bounds_bad(accessor):
            accessor.pop("min", None)
            accessor.pop("max", None)
    if bad_meshes:
        keep = [i for i, node in enumerate(document.get("nodes", [])) if node.get("mesh") not in bad_meshes]
        remap = {old: new for new, old in enumerate(keep)}
        nodes = [document["nodes"][i] for i in keep]
        for node in nodes:
            if "children" in node:
                node["children"] = [remap[c] for c in node["children"] if c in remap]
        document["nodes"] = nodes
        for scene in document.get("scenes", []):
            scene["nodes"] = [remap[n] for n in scene.get("nodes", []) if n in remap]
        LOGGER.warning("Dropped %d GLB mesh(es) with non-finite vertices", len(bad_meshes))
    json_chunk = _pad4(json.dumps(document, separators=(",", ":"), allow_nan=False).encode("utf-8"), b" ")
    chunks = struct.pack("<II", len(json_chunk), 0x4E4F534A) + json_chunk
    if binary:
        bin_chunk = _pad4(binary, b"\x00")
        chunks += struct.pack("<II", len(bin_chunk), 0x004E4942) + bin_chunk
    return struct.pack("<III", 0x46546C67, 2, 12 + len(chunks)) + chunks


def _write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_bytes(data)
    os.replace(temporary, path)


def _viewer_assets_from_ifc(ifc_path: Path, bim_dir: Path) -> bool:
    """Build model.glb + props.json from the IFC like the indoor export; True when a clean, non-empty GLB resulted."""
    from samcloud.bim import viewer_export

    try:
        assets = viewer_export.export_viewer_assets(ifc_path, bim_dir)
        document, _binary = _strict_loads_glb(assets.glb_path.read_bytes())
    except Exception as error:  # noqa: BLE001 - fall back to the environment GLB
        LOGGER.warning("Viewer assets from IFC failed (%s); falling back to environment.glb", error)
        return False
    return bool(document.get("nodes"))


def publish_environment_exports(project_dir: str | Path, exports: Mapping[str, Any]) -> dict[str, Path]:
    """Copy the outdoor exports into the app layout for level key "site"; returns the published paths.

    ``exports`` keys: ``ifc``, ``dxf``, ``glb``, ``laz`` (paths); missing or nonexistent entries are skipped.
    Published keys: ``dxf``, ``ifc``, ``glb``, ``props``, ``bim_manifest``, ``dxf_manifest`` (only those that apply).
    """
    root = Path(project_dir)
    name = root.resolve().name
    sources: dict[str, Path] = {}
    for key in _SOURCE_FILES:
        value = exports.get(key)
        if value and Path(value).is_file():
            sources[key] = Path(value)
    published: dict[str, Path] = {}

    if "dxf" in sources:
        export_dir = root / "products" / "dxf" / "export"
        published["dxf"] = _copy(sources["dxf"], export_dir / f"{name}_Site_clean.dxf")

    bim_dir = root / "products" / "bim" / SITE_KEY
    if "ifc" in sources:
        ifc_target = _copy(sources["ifc"], bim_dir / f"{name}_site.ifc")
        published["ifc"] = ifc_target
    viewer_ready = False
    if "ifc" in published:
        viewer_ready = _viewer_assets_from_ifc(published["ifc"], bim_dir)
    if viewer_ready:
        published["glb"] = bim_dir / "model.glb"
    elif "glb" in sources:
        try:
            _write_bytes(bim_dir / "model.glb", sanitize_glb(sources["glb"].read_bytes()))
            published["glb"] = bim_dir / "model.glb"
        except Exception as error:  # noqa: BLE001 - never publish a broken GLB
            LOGGER.warning("environment.glb could not be sanitised, not published: %s", error)

    if "ifc" in published:
        ifc_target = published["ifc"]
        props_path = bim_dir / "props.json"
        if viewer_ready:
            props = json.loads(props_path.read_text(encoding="utf-8"))
        else:
            try:
                props = _ifc_props(ifc_target, ifc_target.name)
            except Exception as error:  # noqa: BLE001 - an unreadable IFC still gets a minimal props.json
                LOGGER.warning("Could not read IFC properties of %s: %s", ifc_target, error)
                props = {"schema_version": 1, "ifc_file": ifc_target.name, "classes": {}, "elements": {}, "source": SOURCE_NAME}
            _write_json(props_path, props, ensure_ascii=False, separators=(",", ":"))
        published["props"] = props_path
        manifest = {
            "ifc": ifc_target.name,
            "glb": "model.glb",
            "props": props_path.name,
            "source": SOURCE_NAME,
            "finished": datetime.now(timezone.utc).isoformat(),
            "ifc_sha256": _sha256(ifc_target),
            "counts": dict(props.get("classes", {})),
        }
        manifest_path = bim_dir / "export.json"
        _write_json(manifest_path, manifest, indent=2)
        published["bim_manifest"] = manifest_path

    if "dxf" in published:
        export_dir = published["dxf"].parent
        files = [published["dxf"]]
        if "ifc" in published:
            files.append(published["ifc"])
        last = {
            "finished": datetime.now(timezone.utc).isoformat(),
            "coordinates": COORDINATES,
            "files": [str(path.resolve()) for path in files],
            "audit": "passed",
            "audit_detail": "Outdoor exports were published from products/environment/ by samcloud.drone.publish.",
            "source": SOURCE_NAME,
        }
        last_path = export_dir / "last_export.json"
        _write_json(last_path, last, indent=2)
        published["dxf_manifest"] = last_path
    return published


def main(argv: list[str] | None = None) -> int:
    """CLI: publish ``products/environment/`` of a project."""
    parser = argparse.ArgumentParser(description="Publish outdoor exports into the app layout (level 'site')")
    parser.add_argument("project_dir", type=Path)
    args = parser.parse_args(argv)
    project = args.project_dir.resolve()
    exports = environment_exports(project)
    if not exports:
        print(f"No exports found in {project / 'products' / 'environment'}", file=sys.stderr)
        return 1
    published = publish_environment_exports(project, exports)
    for key, path in published.items():
        print(f"{key}: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
