"""Export of the outdoor drone products (LAZ, OBJ, GLB) in LV95 with re-read validation.

Local metric frame -> LV95 (E, N, H LN02) happens only here, at export time.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import laspy
import numpy as np
from pyproj import CRS

from samcloud.drone.contracts import LocalFrame, PointSet
from samcloud.pointcloud.georeference import enu_to_target, load_georeference

logger = logging.getLogger(__name__)

LV95_CRS = "EPSG:2056+5728"
LV95_E_RANGE = (2.48e6, 2.84e6)
LV95_N_RANGE = (1.07e6, 1.30e6)
LAS_SCALE = 0.001


# ----------------------------------------------------------------------------------------
# point cloud
# ----------------------------------------------------------------------------------------

def _to_lv95(xyz: np.ndarray, frame_or_georef: Any) -> np.ndarray:
    """Local points -> LV95 via a LocalFrame or an ENU georeference (dict or JSON path)."""
    xyz = np.asarray(xyz, dtype=np.float64)
    if isinstance(frame_or_georef, LocalFrame):
        return frame_or_georef.to_lv95(xyz)
    georef = frame_or_georef
    if isinstance(georef, (str, Path)):
        georef = load_georeference(Path(georef))
    if isinstance(georef, dict):
        return enu_to_target(xyz, georef, "EPSG:2056")
    raise TypeError("frame_or_georef must be a LocalFrame, a georeference dict or a georeference path")


def export_point_cloud(ps: PointSet, path: Path | str, frame_or_georef: Any, *,
                       extra: dict[str, np.ndarray] | None = None) -> Path:
    """Write a PointSet as LAZ/LAS 1.4 in LV95 (scale 0.001, rounded offsets, CRS VLR).

    Extra dims written when available: confidence, margin (float32), view_count (uint16),
    adjusted (uint8) from the PointSet plus everything in ``extra`` (float -> float32,
    ids such as building_id / tree_id -> int32 so that -1 stays "none", bool -> uint8).
    """
    path = Path(path)
    n = len(ps)
    if n == 0:
        raise ValueError("cannot export an empty point set")
    lv95 = _to_lv95(ps.xyz, frame_or_georef)
    if not np.all(np.isfinite(lv95)):
        raise ValueError("coordinate transformation produced non-finite values")

    has_rgb = ps.rgb is not None
    header = laspy.LasHeader(point_format=7 if has_rgb else 6, version="1.4")
    header.scales = np.array([LAS_SCALE] * 3)
    header.offsets = np.floor(lv95.min(axis=0))
    try:
        header.add_crs(CRS.from_user_input(LV95_CRS))
    except Exception as error:  # pragma: no cover - depends on pyproj/laspy versions
        logger.warning("could not write CRS VLR: %s", error)

    dims: dict[str, np.ndarray] = {}
    if ps.confidence is not None:
        dims["confidence"] = np.asarray(ps.confidence, dtype=np.float32)
    if ps.margin is not None:
        dims["margin"] = np.asarray(ps.margin, dtype=np.float32)
    if ps.view_count is not None:
        dims["view_count"] = np.asarray(ps.view_count).astype(np.uint16)
    if ps.adjusted is not None:
        dims["adjusted"] = np.asarray(ps.adjusted).astype(np.uint8)
    for name, values in (extra or {}).items():
        dims[name] = _cast_extra(name, values, n)
    for name, array in dims.items():
        header.add_extra_dim(laspy.ExtraBytesParams(name=name, type=array.dtype, description=name[:31]))

    las = laspy.LasData(header)
    las.x, las.y, las.z = lv95[:, 0], lv95[:, 1], lv95[:, 2]
    las.classification = np.asarray(ps.classes).astype(np.uint8)
    las.return_number = np.ones(n, dtype=np.uint8)
    las.number_of_returns = np.ones(n, dtype=np.uint8)
    if has_rgb:
        rgb = np.asarray(ps.rgb, dtype=np.uint16) * 257
        las.red, las.green, las.blue = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    for name, array in dims.items():
        setattr(las, name, array)

    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".laz":
        las.write(str(path), do_compress=True, laz_backend=laspy.LazBackend.Lazrs)
    else:
        las.write(str(path))
    return path


def _cast_extra(name: str, values: np.ndarray, n: int) -> np.ndarray:
    array = np.asarray(values)
    if array.shape != (n,):
        raise ValueError(f"extra dim {name!r} needs shape ({n},), got {array.shape}")
    if array.dtype == np.bool_:
        return array.astype(np.uint8)
    if array.dtype.kind == "f":
        return array.astype(np.float32)
    if array.dtype.kind in "iu":
        if name.endswith("_id"):
            return array.astype(np.int32)
        return array.astype(np.uint16 if array.dtype.itemsize <= 2 else np.uint32)
    raise ValueError(f"extra dim {name!r}: unsupported dtype {array.dtype}")


# ----------------------------------------------------------------------------------------
# meshes
# ----------------------------------------------------------------------------------------

def _check_mesh(vertices: np.ndarray, faces: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    v = np.asarray(vertices, dtype=np.float64)
    f = np.asarray(faces, dtype=np.int64)
    if v.ndim != 2 or v.shape[1] != 3 or f.ndim != 2 or f.shape[1] != 3 or len(v) == 0 or len(f) == 0:
        raise ValueError("mesh needs (V,3) vertices and (F,3) faces, both non-empty")
    if f.min() < 0 or f.max() >= len(v):
        raise ValueError("face index out of range")
    return v, f


def export_mesh(vertices: np.ndarray, faces: np.ndarray, path: Path | str, frame: LocalFrame, *,
                fmt: str = "obj", node_name: str | None = None) -> Path:
    """Write one mesh. OBJ: LV95 coordinates (3 decimals); GLB: local coordinates + extras.origin_lv95."""
    v, f = _check_mesh(vertices, faces)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    name = node_name or path.stem
    if fmt == "obj":
        lv = frame.to_lv95(v)
        lines = [f"# CRS: EPSG:{frame.epsg} + {frame.vertical_datum} (E N H, metres)",
                 f"o {name}"]
        lines += [f"v {x:.3f} {y:.3f} {z:.3f}" for x, y, z in lv]
        lines += [f"f {a + 1} {b + 1} {c + 1}" for a, b, c in f]
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    elif fmt == "glb":
        _write_glb({name: (v, f)}, path, frame)
    else:
        raise ValueError(f"unknown mesh format {fmt!r} (obj or glb)")
    return path


def export_scene_glb(nodes: dict[str, tuple[np.ndarray, np.ndarray]], path: Path | str,
                     frame: LocalFrame) -> Path:
    """One GLB, one named node per entry (local coordinates, extras.origin_lv95)."""
    if not nodes:
        raise ValueError("no nodes to export")
    checked = {name: _check_mesh(v, f) for name, (v, f) in nodes.items()}
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    _write_glb(checked, path, frame)
    return path


def _write_glb(nodes: dict[str, tuple[np.ndarray, np.ndarray]], path: Path, frame: LocalFrame) -> None:
    import trimesh

    scene = trimesh.Scene()
    for name, (v, f) in nodes.items():
        scene.add_geometry(trimesh.Trimesh(vertices=v, faces=f, process=False),
                           node_name=name, geom_name=name)
    scene.metadata["origin_lv95"] = [float(c) for c in frame.origin_lv95]
    scene.metadata["epsg"] = int(frame.epsg)
    scene.metadata["vertical_datum"] = frame.vertical_datum
    path.write_bytes(scene.export(file_type="glb"))


# ----------------------------------------------------------------------------------------
# validation
# ----------------------------------------------------------------------------------------

def validate_exports(paths: dict[str, Path | str]) -> dict[str, dict]:
    """Re-read every file and report ``{"ok": bool, "reason": str, ...details}`` per key."""
    handlers = {".las": _check_las, ".laz": _check_las, ".tif": _check_tif, ".tiff": _check_tif,
                ".obj": _check_mesh_file, ".glb": _check_mesh_file, ".ifc": _check_ifc,
                ".dxf": _check_dxf}
    report: dict[str, dict] = {}
    for key, raw in paths.items():
        path = Path(raw)
        if not path.is_file():
            report[key] = {"ok": False, "reason": "file missing"}
            continue
        if path.stat().st_size == 0:
            report[key] = {"ok": False, "reason": "file is empty"}
            continue
        handler = handlers.get(path.suffix.lower())
        if handler is None:
            report[key] = {"ok": False, "reason": f"unsupported extension {path.suffix}"}
            continue
        try:
            report[key] = handler(path)
        except Exception as error:  # a broken file must be reported, not raised
            report[key] = {"ok": False, "reason": f"unreadable: {type(error).__name__}: {error}"}
    return report


def _check_las(path: Path) -> dict:
    las = laspy.read(str(path))
    count = int(len(las.points))
    dims = list(las.point_format.extra_dimension_names)
    if count == 0:
        return {"ok": False, "reason": "no points", "points": 0, "extra_dims": dims}
    x, y = np.asarray(las.x), np.asarray(las.y)
    bbox = [float(x.min()), float(y.min()), float(x.max()), float(y.max())]
    inside = (LV95_E_RANGE[0] <= bbox[0] and bbox[2] <= LV95_E_RANGE[1]
              and LV95_N_RANGE[0] <= bbox[1] and bbox[3] <= LV95_N_RANGE[1])
    return {"ok": bool(inside), "reason": "ok" if inside else "bbox outside LV95 range",
            "points": count, "extra_dims": dims, "bbox": bbox}


def _check_tif(path: Path) -> dict:
    import rasterio

    with rasterio.open(path) as src:
        epsg = src.crs.to_epsg() if src.crs else None
        data = src.read(masked=True)
        finite = bool(data.count() > 0 and np.all(np.isfinite(data.compressed())))
        info = {"epsg": epsg, "shape": list(data.shape), "valid_pixels": int(data.count())}
        if epsg != 2056:
            return {"ok": False, "reason": f"CRS is {epsg}, expected 2056", **info}
        if not finite:
            return {"ok": False, "reason": "no finite pixel values", **info}
        values = data.compressed().astype(np.float64)
        return {"ok": True, "reason": "ok", "min": float(values.min()), "max": float(values.max()),
                "mean": float(values.mean()), **info}


def _check_mesh_file(path: Path) -> dict:
    import trimesh

    kind = path.suffix.lower().lstrip(".")
    loaded = trimesh.load(str(path), file_type=kind, process=False, force="scene")
    meshes = [g for g in loaded.geometry.values() if hasattr(g, "faces")]
    vertices = int(sum(len(m.vertices) for m in meshes))
    faces = int(sum(len(m.faces) for m in meshes))
    watertight = bool(meshes) and all(m.is_watertight for m in meshes)
    info = {"vertices": vertices, "faces": faces, "watertight": watertight, "nodes": len(meshes)}
    if vertices == 0 or faces == 0:
        return {"ok": False, "reason": "no vertices/faces", **info}
    return {"ok": True, "reason": "ok", **info}


def _check_ifc(path: Path) -> dict:
    import ifcopenshell

    model = ifcopenshell.open(str(path))
    schema = model.schema
    conversions = model.by_type("IfcMapConversion")
    info = {"schema": schema, "map_conversions": len(conversions)}
    if schema != "IFC4":
        return {"ok": False, "reason": f"schema is {schema}, expected IFC4", **info}
    if not conversions:
        return {"ok": False, "reason": "IfcMapConversion missing", **info}
    return {"ok": True, "reason": "ok", **info}


def _check_dxf(path: Path) -> dict:
    import ezdxf

    doc = ezdxf.readfile(str(path))
    count = len(doc.modelspace())
    return {"ok": count > 0, "reason": "ok" if count else "modelspace is empty", "entities": count}


def lv95_sample_check(ps_local: PointSet | np.ndarray, laz_path: Path | str, frame: LocalFrame,
                      n: int = 20, *, seed: int = 0, tol_m: float = 1e-3) -> dict:
    """Compare ``frame.to_lv95(local)`` of random points with the nearest points in the file.

    Passes when the largest per-axis deviation is below ``tol_m`` (1 mm).
    """
    from scipy.spatial import cKDTree

    xyz = np.asarray(ps_local.xyz if isinstance(ps_local, PointSet) else ps_local, dtype=np.float64)
    if len(xyz) == 0:
        return {"ok": False, "reason": "no local points", "n": 0, "max_err_m": float("nan")}
    rng = np.random.default_rng(seed)
    pick = rng.choice(len(xyz), size=min(n, len(xyz)), replace=False)
    expected = frame.to_lv95(xyz[pick])
    las = laspy.read(str(laz_path))
    stored = np.column_stack((np.asarray(las.x), np.asarray(las.y), np.asarray(las.z)))
    if len(stored) == 0:
        return {"ok": False, "reason": "file has no points", "n": 0, "max_err_m": float("nan")}
    _, nearest = cKDTree(stored).query(expected)
    err = float(np.abs(stored[nearest] - expected).max())
    ok = err < tol_m
    return {"ok": bool(ok), "reason": "ok" if ok else f"max deviation {err:.4f} m >= {tol_m} m",
            "n": int(len(pick)), "max_err_m": err}


# ----------------------------------------------------------------------------------------
# clipped copy of the classified cloud for the viewer (raw classified.ply stays untouched)
# ----------------------------------------------------------------------------------------

CLIPPED_PLY_NAME = "classified_clipped.ply"


def ensure_clipped_classified_ply(project_dir: Path | str, config: dict | None = None) -> Path | None:
    """Boundary-clipped copy ``products/classified/classified_clipped.ply`` for outdoor/drone projects.

    ``classified.ply`` (COLMAP ENU frame) is the raw pipeline output and is only read. The copy keeps all vertex
    properties; the boundary (LV95, ``products/environment/boundary.geojson``) is applied through the project
    georeference, so the ENU/LV95 rotation is honoured. Rebuilt when the source, the boundary or the georeference
    changes. Returns ``None`` when no clipping applies (indoor, switch off, no boundary/georeference, errors).
    """
    import hashlib

    from samcloud.drone import frames, perimeter
    from samcloud.engine.export_las import clip_ply

    project = Path(project_dir)
    polygon = perimeter.clip_polygon_lv95(project, config)
    source = project / "products" / "classified" / "classified.ply"
    if polygon is None or not source.is_file() or source.stat().st_size == 0:
        return None
    target = source.with_name(CLIPPED_PLY_NAME)
    sidecar = target.with_suffix(".json")
    boundary_file = project / perimeter.BOUNDARY_GEOJSON
    georef_file = project / "products" / "georeference.json"
    signature = {
        "source_size": source.stat().st_size, "source_mtime_ns": source.stat().st_mtime_ns,
        "boundary": hashlib.sha1(boundary_file.read_bytes()).hexdigest(),
        "georeference": hashlib.sha1(georef_file.read_bytes()).hexdigest() if georef_file.is_file() else None,
    }
    try:
        if target.is_file() and sidecar.is_file() and json.loads(sidecar.read_text(encoding="utf-8")).get("signature") == signature:
            return target
        georef = frames.project_georeference(project)
        before, after = clip_ply(source, target, lambda xyz: perimeter.inside_mask_enu(xyz, georef, polygon))
    except Exception as error:  # the viewer then shows the unclipped cloud rather than failing
        logger.warning("could not clip %s to the processing boundary: %s", source, error)
        return None
    sidecar.write_text(json.dumps({"signature": signature, "points_before": before, "points_after": after}), encoding="utf-8")
    logger.info("clipped classified cloud for the viewer: %d -> %d points", before, after)
    return target
