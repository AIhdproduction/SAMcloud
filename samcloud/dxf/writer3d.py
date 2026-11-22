"""3D DXF of the written IFC: every product tessellated (ifcopenshell.geom) and written as one MESH entity.

The 3D DXF always matches the IFC (walls, slabs, openings filled by doors / windows, furniture with its reconstructed
shape, and whatever other exporters add: columns, beams, stairs ...).  One layer per IFC class (``3D-IfcWall``,
``3D-IfcFurniture`` ...); each MESH carries the product's GlobalId, class and name as XDATA (application ``SAMCLOUD``).
Coordinates are the IFC world coordinates in metres (``$INSUNITS`` = 6) unless ``scale`` is given.  Voids
(IfcOpeningElement) and, by default, spaces are not written.  The file is saved atomically after an ezdxf audit.
"""

from __future__ import annotations

import logging
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import ezdxf
import numpy as np

_LOGGER = logging.getLogger(__name__)

LAYER_PREFIX = "3D-"
APP_ID = "SAMCLOUD"
SKIPPED_CLASSES = ("IfcOpeningElement", "IfcVirtualElement", "IfcAnnotation", "IfcGrid")

# AutoCAD colour index per IFC class (layer colour); others take 7 (white / black)
_LAYER_COLOURS: dict[str, int] = {
    "IfcWall": 8, "IfcWallStandardCase": 8, "IfcSlab": 9, "IfcCovering": 254, "IfcRoof": 9, "IfcColumn": 252, "IfcBeam": 252,
    "IfcStair": 33, "IfcStairFlight": 33, "IfcSlabLanding": 33, "IfcRailing": 251, "IfcDoor": 30, "IfcWindow": 4,
    "IfcFurniture": 42, "IfcSanitaryTerminal": 5, "IfcElectricAppliance": 6, "IfcLightFixture": 2, "IfcAudioVisualAppliance": 250,
    "IfcSpaceHeater": 1, "IfcBuildingElementProxy": 3, "IfcSpace": 151,
    "IfcSensor": 20, "IfcAirTerminal": 140, "IfcFireSuppressionTerminal": 10,
}


@dataclass(slots=True)
class Dxf3dResult:
    path: Path
    meshes: int = 0
    faces: int = 0
    layers: dict[str, int] = field(default_factory=dict)
    skipped: list[str] = field(default_factory=list)


def layer_name(ifc_class: str) -> str:
    return f"{LAYER_PREFIX}{ifc_class}"


def _products(model: Any, include_spaces: bool) -> list[Any]:
    products = []
    for product in model.by_type("IfcProduct"):
        if product.Representation is None or any(product.is_a(name) for name in SKIPPED_CLASSES):
            continue
        if product.is_a("IfcSpace") and not include_spaces:
            continue
        products.append(product)
    return products


def _tessellate(model: Any, products: list[Any]) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    import ifcopenshell.geom

    settings = ifcopenshell.geom.settings()
    settings.set("use-world-coords", True)
    shapes: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    if not products:
        return shapes
    iterator = ifcopenshell.geom.iterator(settings, model, max(1, min(8, os.cpu_count() or 1)), include=products)
    if iterator.initialize():
        while True:
            shape = iterator.get()
            verts = np.asarray(shape.geometry.verts, dtype=np.float64).reshape(-1, 3)
            faces = np.asarray(shape.geometry.faces, dtype=np.int64).reshape(-1, 3)
            if len(verts) and len(faces):
                shapes[int(shape.id)] = (verts, faces)
            if not iterator.next():
                break
    return shapes


def _weld(verts: np.ndarray, faces: np.ndarray, digits: int = 6) -> tuple[np.ndarray, np.ndarray]:
    """Merge coincident vertices and drop degenerate faces (a clean MESH for CAD)."""
    unique, inverse = np.unique(np.round(verts, digits), axis=0, return_inverse=True)
    faces = inverse.reshape(-1)[faces]
    keep = (faces[:, 0] != faces[:, 1]) & (faces[:, 1] != faces[:, 2]) & (faces[:, 0] != faces[:, 2])
    return unique, faces[keep]


def build_dxf3d_document(ifc_path: Path | str, *, include_spaces: bool = False, scale: float = 1.0) -> tuple[ezdxf.document.Drawing, Dxf3dResult]:
    import ifcopenshell

    model = ifcopenshell.open(str(ifc_path))
    products = _products(model, include_spaces)
    shapes = _tessellate(model, products)
    document = ezdxf.new("R2018", setup=True)
    document.header["$INSUNITS"] = 6
    if APP_ID not in document.appids:
        document.appids.new(APP_ID)
    msp = document.modelspace()
    result = Dxf3dResult(Path(ifc_path))
    for product in products:
        shape = shapes.get(int(product.id()))
        ifc_class = product.is_a()
        if shape is None:
            result.skipped.append(f"{ifc_class} {product.GlobalId}: no geometry")
            continue
        verts, faces = _weld(*shape)
        if len(faces) == 0:
            result.skipped.append(f"{ifc_class} {product.GlobalId}: degenerate geometry")
            continue
        layer = layer_name(ifc_class)
        if layer not in document.layers:
            document.layers.add(layer, color=_LAYER_COLOURS.get(ifc_class, 7))
        mesh = msp.add_mesh(dxfattribs={"layer": layer})
        with mesh.edit_data() as data:
            data.vertices = [tuple(float(v) * scale for v in vertex) for vertex in verts]
            data.faces = [tuple(int(i) for i in face) for face in faces]
        mesh.set_xdata(APP_ID, [(1000, str(product.GlobalId)), (1000, ifc_class), (1000, str(product.Name or ""))])
        result.meshes += 1
        result.faces += int(len(faces))
        result.layers[layer] = result.layers.get(layer, 0) + 1
    return document, result


def export_ifc_3d_dxf(ifc_path: Path | str, destination: Path | str, *, include_spaces: bool = False, scale: float = 1.0) -> Dxf3dResult:
    """Write the 3D DXF of ``ifc_path`` to ``destination`` (atomic; raises when the saved file does not audit clean)."""
    document, result = build_dxf3d_document(ifc_path, include_spaces=include_spaces, scale=scale)
    target = Path(destination)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{target.stem}.", suffix=".tmp", dir=target.parent)
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        document.saveas(temporary)
        auditor = ezdxf.readfile(temporary).audit()
        if auditor.has_errors:
            raise RuntimeError(f"3D DXF audit found {len(auditor.errors)} error(s)")
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            temporary.unlink()
    result.path = target
    return result


def dxf3d_path(project_dir: Path | str, project_name: str, level_name: str) -> Path:
    """``products/dxf/export/<project>_<level>_3d.dxf`` (names sanitised like the 2D export)."""
    from samcloud.dxf.writer import sanitize_dxf_name

    return Path(project_dir) / "products" / "dxf" / "export" / f"{sanitize_dxf_name(project_name)}_{sanitize_dxf_name(level_name)}_3d.dxf"


def export_level_3d_dxf(project_dir: Path | str, level_id: str, level_name: str, logger: logging.Logger | None = None) -> Path | None:
    """run_all hook after "Export IFC": 3D DXF of the level's IFC.  Never raises; a failure is logged as a warning."""
    log = logger or _LOGGER
    try:
        import json

        bim_dir = Path(project_dir) / "products" / "bim" / str(level_id)
        manifest = json.loads((bim_dir / "export.json").read_text(encoding="utf-8"))
        ifc_path = bim_dir / str(manifest["ifc"])
        target = dxf3d_path(project_dir, Path(project_dir).name, level_name)
        result = export_ifc_3d_dxf(ifc_path, target)
        log.info(f"3D DXF exported: {target} ({result.meshes} meshes, {result.faces} faces, {len(result.layers)} layers)")
        return target
    except Exception as error:  # noqa: BLE001 - the 3D DXF is an extra product; the run must not fail
        log.warning(f"3D DXF export skipped: {type(error).__name__}: {error}")
        return None


def dxf3d_is_current(project_dir: Path | str, level_id: str, level_name: str) -> bool:
    """False only when the level has an IFC and its 3D DXF is missing or older than it (then the IFC step reruns)."""
    target = dxf3d_path(project_dir, Path(project_dir).name, level_name)
    bim_dir = Path(project_dir) / "products" / "bim" / str(level_id)
    ifc_files = list(bim_dir.glob("*.ifc"))
    if not ifc_files:
        return True
    if not target.is_file():
        return False
    return target.stat().st_mtime_ns >= max(path.stat().st_mtime_ns for path in ifc_files)
