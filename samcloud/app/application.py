"""Route registration for the NiceGUI application."""

import json
import struct
from pathlib import Path

import numpy as np

from nicegui import app, ui
from starlette.responses import FileResponse, JSONResponse, Response

from samcloud.app.pages import control_points, dense_cloud, dxf, home, projects, settings, sparse_cloud, targets
from samcloud.app.i18n import t
from samcloud.app.state import state
from samcloud.app.dxf_controller import DxfPageController
from samcloud.app.dxf_review_canvas import review_background, review_payload
from samcloud.bim import ui_data as bim_ui_data
from samcloud.app.gcp_review import apriltag_marker_payload, apriltag_warnings, image_crop, marker_payload, resolve_project_image
from samcloud.core.paths import ASSETS_DIRECTORY
from samcloud.control_points.repository import (
    move_detection_observation,
    remove_detection_observation,
    set_control_point_role,
    set_detection_enabled,
    set_detection_observation_enabled,
    set_detection_role,
)
from samcloud.viewer.preview import _ply_header_size, _read_ply_header, ensure_preview
from samcloud.viewer.potree import ensure_potree, potree_status, start_potree_build
from samcloud.viewer.cameras import camera_payload, camera_thumbnail


_INSTANCE_POINT_LIMIT = 200_000


def _instances_path(project_directory: Path) -> Path:
    return project_directory / "products" / "classified" / "instances.json"


def _classified_ply_path(project_directory: Path) -> Path:
    return project_directory / "products" / "classified" / "classified.ply"


def _instance_class_config(project_directory: Path) -> dict:
    """Load the project class config for German instance/class labels; empty when unavailable."""
    try:
        from samcloud.classification.custom import load_project_classes

        values = load_project_classes(project_directory)
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        values = {}
    return values if isinstance(values, dict) else {}


def _instance_label_map(project_directory: Path) -> dict[str, str]:
    """Load optional German instance labels without requiring a newer classes config."""
    values = _instance_class_config(project_directory).get("instance_labels_de", {})
    return values if isinstance(values, dict) else {}


def _class_names_de(config: dict) -> list:
    """German class names of the active project's class set (index = class id), or an empty list."""
    try:
        project = state.active_project()
        class_set = (project or {}).get("settings", {}).get("classification", {}).get("class_set", "outdoor")
    except Exception:
        class_set = "outdoor"
    labels = config.get("labels_de", {})
    names = labels.get(class_set) if isinstance(labels, dict) else None
    return names if isinstance(names, list) else []


def _instance_payload(project_directory: Path) -> list[dict[str, object]]:
    """Return the public instance metadata, or no instances for older projects."""
    path = _instances_path(project_directory)
    if not path.is_file():
        return []
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    class_config = _instance_class_config(project_directory)
    labels_de = class_config.get("instance_labels_de", {})
    labels_de = labels_de if isinstance(labels_de, dict) else {}
    class_names_de = _class_names_de(class_config)
    result: list[dict[str, object]] = []
    for item in document.get("instances", []) if isinstance(document, dict) else []:
        if not isinstance(item, dict) or not isinstance(item.get("id"), int):
            continue
        label = str(item.get("label") or item.get("class_name") or "Object")
        class_id = item.get("class_id")
        class_de = class_names_de[class_id] if isinstance(class_id, int) and 0 <= class_id < len(class_names_de) else None
        result.append({
            "id": item["id"],
            "label": label,
            "label_de": str(labels_de.get(label) or class_de or label),
            "label_index": item.get("label_index"),
            "class_id": item.get("class_id"),
            "class_name": item.get("class_name"),
            "point_count": item.get("point_count", 0),
            "bbox_min": item.get("bbox_min"),
            "bbox_max": item.get("bbox_max"),
        })
    return result


def _instance_index(project_directory: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.dtype, dict[str, int]]:
    """Load or rebuild the cached stable ordering of PLY rows by non-zero instance id."""
    source = _classified_ply_path(project_directory)
    if not source.is_file():
        raise FileNotFoundError(t("app.errors.no_classified_cloud"))
    cache_path = project_directory / "products" / "viewer" / "instance_index.npz"
    source_stat = source.stat()
    signature = (source_stat.st_mtime_ns, source_stat.st_size)
    try:
        with np.load(cache_path) as cache:
            if tuple(int(value) for value in cache["signature"]) == signature:
                count, _properties, dtype, offsets = _read_ply_header(source)
                return cache["ids"], cache["offsets"], cache["indices"], dtype, offsets
    except (OSError, ValueError, KeyError):
        pass
    count, _properties, dtype, fields = _read_ply_header(source)
    if "instance_id" not in fields:
        return np.empty(0, np.uint32), np.zeros(1, np.int64), np.empty(0, np.int64), dtype, fields
    data = np.fromfile(source, dtype=dtype, count=count, offset=_ply_header_size(source))
    instance_ids = np.asarray(data["instance_id"], dtype=np.uint32)
    valid_indices = np.flatnonzero(instance_ids)
    ordered_indices = valid_indices[np.argsort(instance_ids[valid_indices], kind="stable")]
    ordered_ids = instance_ids[ordered_indices]
    ids, starts = np.unique(ordered_ids, return_index=True)
    boundaries = np.append(starts, len(ordered_indices)).astype(np.int64)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        signature=np.asarray(signature, dtype=np.int64),
        ids=ids,
        offsets=boundaries,
        indices=ordered_indices.astype(np.int64),
    )
    return ids, boundaries, ordered_indices.astype(np.int64), dtype, fields


@app.get("/api/instances")
def instances():
    """Serve recognized dense-cloud object instances for the current project."""
    project_directory = state.active_project_directory
    if project_directory is None:
        return JSONResponse({"error": t("app.errors.no_project")}, status_code=404)
    return JSONResponse({"instances": _instance_payload(project_directory)})


@app.get("/api/instances/{instance_id}/points")
def instance_points(instance_id: int):
    """Serve one instance's original-colour PLY points in the Potree coordinate frame."""
    project_directory = state.active_project_directory
    if project_directory is None:
        return JSONResponse({"error": t("app.errors.no_project")}, status_code=404)
    try:
        ids, offsets, indices, dtype, fields = _instance_index(project_directory)
        position = int(np.searchsorted(ids, instance_id))
        if position >= len(ids) or int(ids[position]) != instance_id:
            return JSONResponse({"error": t("app.errors.instance_not_found")}, status_code=404)
        selected = indices[offsets[position]:offsets[position + 1]]
        if len(selected) > _INSTANCE_POINT_LIMIT:
            selected = selected[np.linspace(0, len(selected) - 1, _INSTANCE_POINT_LIMIT, dtype=np.int64)]
        source = _classified_ply_path(project_directory)
        count = _read_ply_header(source)[0]
        # Memory-map and gather only the requested rows; the cloud is never read in full.
        rows = np.memmap(source, dtype=dtype, mode="r", offset=_ply_header_size(source), shape=(count,))
        data = np.asarray(rows[np.sort(selected)])
        del rows
    except (FileNotFoundError, OSError, ValueError) as error:
        return JSONResponse({"error": str(error)}, status_code=404)
    record = np.empty(len(data), dtype=np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                                                  ("r", "u1"), ("g", "u1"), ("b", "u1")]))
    record["x"], record["y"], record["z"] = data["x"], data["y"], data["z"]
    for channel, name in (("r", "red"), ("g", "green"), ("b", "blue")):
        record[channel] = data[name] if name in fields else 200
    records = b"SAMV" + struct.pack("<III", 1, len(data), 0) + record.tobytes()
    return Response(records, media_type="application/octet-stream", headers={"Cache-Control": "no-store"})


@app.get("/api/pointcloud/{kind}")
def pointcloud_preview(kind: str):
    """Serve a compact, browser-friendly preview of the active project's cloud."""
    project_directory = state.active_project_directory
    if project_directory is None:
        return JSONResponse({"error": t("app.errors.no_project")}, status_code=404)
    try:
        preview_path, point_count = ensure_preview(project_directory, kind)
    except (FileNotFoundError, ValueError, OSError) as error:
        return JSONResponse({"error": str(error)}, status_code=404)
    return FileResponse(
        preview_path,
        media_type="application/octet-stream",
        headers={
            "Cache-Control": "no-store",
            "X-SAMcloud-Point-Count": str(point_count),
        },
    )


@app.get("/api/cameras/sparse")
def sparse_cameras():
    """Serve final-model camera centres for the sparse-cloud inspector."""
    project_directory = state.active_project_directory
    if project_directory is None:
        return JSONResponse({"error": t("app.errors.no_project")}, status_code=404)
    try:
        return JSONResponse(camera_payload(project_directory))
    except (FileNotFoundError, OSError, ValueError) as error:
        return JSONResponse({"error": str(error)}, status_code=404)


@app.get("/api/camera-thumbnail/{camera_id}")
def sparse_camera_thumbnail(camera_id: str, size: str = "320"):
    """Serve a downscaled, ID-addressed source image for a sparse camera."""
    project_directory = state.active_project_directory
    if project_directory is None:
        return JSONResponse({"error": t("app.errors.no_project")}, status_code=404)
    try:
        encoded = camera_thumbnail(project_directory, camera_id, size)
    except ValueError as error:
        return JSONResponse({"error": str(error)}, status_code=400)
    except FileNotFoundError as error:
        return JSONResponse({"error": str(error)}, status_code=404)
    except OSError as error:
        return JSONResponse({"error": str(error)}, status_code=500)
    return Response(encoded, media_type="image/jpeg", headers={"Cache-Control": "private, max-age=3600"})


@app.get("/api/potree/{kind}/status")
def potree_build_status(kind: str):
    """Start an octree build when needed and return its current status."""
    project_directory = state.active_project_directory
    if project_directory is None:
        return JSONResponse({"error": t("app.errors.no_project")}, status_code=404)
    try:
        start_potree_build(project_directory, kind)
        return JSONResponse(potree_status(project_directory, kind))
    except ValueError as error:
        return JSONResponse({"error": str(error)}, status_code=404)


@app.get("/api/potree/{kind}/{file_path:path}")
def potree_file(kind: str, file_path: str):
    """Serve Potree octree files; converts the active project's cloud on the first metadata request."""
    project_directory = state.active_project_directory
    if project_directory is None:
        return JSONResponse({"error": t("app.errors.no_project")}, status_code=404)
    try:
        root = ensure_potree(project_directory, kind) if file_path == "metadata.json" else project_directory / "products" / "viewer" / "potree" / kind
    except (FileNotFoundError, ValueError) as error:
        return JSONResponse({"error": str(error)}, status_code=404)
    except RuntimeError as error:
        return JSONResponse({"error": str(error)}, status_code=500)
    root = root.resolve()
    candidate = (root / file_path).resolve()
    try:
        candidate.relative_to(root)
    except ValueError:
        return JSONResponse({"error": t("app.errors.invalid_path")}, status_code=400)
    if not candidate.is_file():
        return JSONResponse({"error": t("app.errors.file_not_found")}, status_code=404)
    return FileResponse(candidate, headers={"Cache-Control": "no-store"})


@app.get("/api/control-point-image/{image_path:path}")
def control_point_image(image_path: str):
    """Serve only a file below the active project's dense image directory."""
    project_directory = state.active_project_directory
    if project_directory is None:
        return JSONResponse({"error": t("app.errors.no_project")}, status_code=404)
    image_root = (project_directory / "products" / "dense" / "images").resolve()
    if not image_root.is_dir():
        try:
            return FileResponse(
                resolve_project_image(project_directory, image_path, "original"),
                headers={"Cache-Control": "no-store"},
            )
        except (FileNotFoundError, ValueError):
            return JSONResponse({"error": t("app.errors.image_not_found")}, status_code=404)
    candidate = (image_root / image_path).resolve()
    try:
        candidate.relative_to(image_root)
    except ValueError:
        return JSONResponse({"error": t("app.errors.invalid_image_path")}, status_code=400)
    if not candidate.is_file():
        return JSONResponse({"error": t("app.errors.image_not_found")}, status_code=404)
    return FileResponse(candidate, headers={"Cache-Control": "no-store"})


@app.get("/api/control-points/markers")
def control_point_markers():
    """Serve reviewed markers, errors, and final-model coordinates for the sparse view."""
    project_directory = state.active_project_directory
    if project_directory is None:
        return JSONResponse({"error": t("app.errors.no_project")}, status_code=404)
    return JSONResponse(marker_payload(project_directory))


@app.get("/api/apriltags/markers")
def apriltag_markers():
    """Serve the AprilTag squares (final sparse-model frame = dense viewer frame) for the dense view."""
    project_directory = state.active_project_directory
    if project_directory is None:
        return JSONResponse({"markers": [], "warnings": []})
    return JSONResponse({
        "markers": apriltag_marker_payload(project_directory),
        "warnings": apriltag_warnings(project_directory),
    })


@app.get("/api/control-points/crop")
def control_point_crop(image: str, u: str, v: str, size: str = "96", source: str = "original"):
    """Serve a small, path-safe original or undistorted marker crop."""
    project_directory = state.active_project_directory
    if project_directory is None:
        return JSONResponse({"error": t("app.errors.no_project")}, status_code=404)
    try:
        encoded, crop_size, origin_u, origin_v = image_crop(project_directory, image, u, v, size, source)
    except ValueError as error:
        return JSONResponse({"error": str(error)}, status_code=400)
    except FileNotFoundError as error:
        return JSONResponse({"error": str(error)}, status_code=404)
    except OSError as error:
        return JSONResponse({"error": str(error)}, status_code=500)
    return Response(
        encoded,
        media_type="image/png",
        headers={
            "Cache-Control": "no-store",
            "X-SAMcloud-Crop-Size": str(crop_size),
            "X-Crop-Origin-U": str(origin_u),
            "X-Crop-Origin-V": str(origin_v),
            "X-Crop-Size": str(crop_size),
        },
    )


@app.post("/api/control-points/observation")
def move_control_point_observation(payload: dict[str, object]):
    """Store one manual original-image marker placement."""
    return _review_action(payload, lambda directory, value: move_detection_observation(
        directory / "control_points" / "detections.json", _required_text(value, "name"),
        _required_text(value, "image"), value.get("u"), value.get("v"),
    ))


@app.post("/api/control-points/observation/remove")
def remove_control_point_observation(payload: dict[str, object]):
    """Remove a reviewed marker projection."""
    return _review_action(payload, lambda directory, value: remove_detection_observation(
        directory / "control_points" / "detections.json", _required_text(value, "name"), _required_text(value, "image"),
    ))


@app.post("/api/control-points/observation/toggle")
def toggle_control_point_observation(payload: dict[str, object]):
    """Persist whether an individual projection participates in adjustment."""
    return _review_action(payload, lambda directory, value: set_detection_observation_enabled(
        directory / "control_points" / "detections.json", _required_text(value, "name"),
        _required_text(value, "image"), value.get("enabled"),
    ))


@app.post("/api/control-points/marker/toggle")
def toggle_control_point_marker(payload: dict[str, object]):
    """Persist whether a marker is used in the next GCP adjustment."""
    return _review_action(payload, lambda directory, value: set_detection_enabled(
        directory / "control_points" / "detections.json", _required_text(value, "name"), value.get("enabled"),
    ))


@app.post("/api/control-points/marker/role")
def change_control_point_role(payload: dict[str, object]):
    """Change a marker's pass/control role in both persisted review inputs."""
    def action(directory, value):
        name, role = _required_text(value, "name"), value.get("role")
        set_detection_role(directory / "control_points" / "detections.json", name, role)
        csv_path = directory / "control_points" / "control_points.csv"
        if csv_path.is_file():
            set_control_point_role(csv_path, name, role)
        return {"role": role}
    return _review_action(payload, action)


@app.post("/api/control-points/optimize")
def optimize_reviewed_control_points(payload: dict[str, object] | None = None):
    """Re-run only the GCP adjustment in a background thread."""
    project_directory = state.active_project_directory
    if project_directory is None:
        return JSONResponse({"error": t("app.errors.no_project")}, status_code=404)
    try:
        state.processing.optimize_control_points(project_directory)
    except (OSError, ValueError) as error:
        return JSONResponse({"error": str(error)}, status_code=409)
    return JSONResponse({"status": "started"}, status_code=202)


@app.post("/api/control-points/continue")
def continue_reviewed_control_points(payload: dict[str, object] | None = None):
    """Continue the existing dense-processing phase after a successful review."""
    project_directory = state.active_project_directory
    if project_directory is None:
        return JSONResponse({"error": t("app.errors.no_project")}, status_code=404)
    try:
        state.processing.continue_after_control_points(project_directory)
    except (OSError, ValueError) as error:
        return JSONResponse({"error": str(error)}, status_code=409)
    return JSONResponse({"status": "started"}, status_code=202)


@app.get("/api/dxf/payload/{level_key}")
def dxf_review_payload(level_key: str, variant: str = "pointcloud"):
    """Serve the cached DXF vector payload of a level ('site' for outdoor); never generates."""
    project_directory = state.active_project_directory
    if project_directory is None:
        return JSONResponse({"error": t("app.errors.no_project")}, status_code=404)
    try:
        if variant not in {"pointcloud", "ortho"}:
            raise ValueError(t("app.errors.invalid_plan_variant"))
        payload = review_payload(project_directory, level_key, variant)
    except ValueError as error:
        return JSONResponse({"error": str(error)}, status_code=400)
    if payload is None:
        return JSONResponse({"error": t("app.errors.no_dxf_preview")}, status_code=404)
    return JSONResponse(payload, headers={"Cache-Control": "no-store"})


@app.get("/api/dxf/background/{level_key}/{source}")
def dxf_review_background(level_key: str, source: str):
    """Serve the cached north-up background PNG ('rgb' or 'classes') of a level; never generates."""
    project_directory = state.active_project_directory
    if project_directory is None:
        return JSONResponse({"error": t("app.errors.no_project")}, status_code=404)
    try:
        path = review_background(project_directory, level_key, source)
    except ValueError as error:
        return JSONResponse({"error": str(error)}, status_code=400)
    if path is None:
        return JSONResponse({"error": t("app.errors.no_background")}, status_code=404)
    return FileResponse(path, media_type="image/png", headers={"Cache-Control": "private, max-age=3600"})


ORTHO_MEDIA_TYPES: dict[str, str] = {
    "preview": "image/jpeg",
    "jpg8000": "image/jpeg",
    "jpg4096": "image/jpeg",
    "png": "image/png",
    "pgw": "text/plain",
    "json": "application/json",
}


@app.get("/api/dxf/ortho/{level_key}/{kind}")
def dxf_ortho_file(level_key: str, kind: str, v: str | None = None):
    """Serve a cached orthophoto artifact; reads only the cache (never creates or generates)."""
    project_directory = state.active_project_directory
    if project_directory is None:
        return JSONResponse({"error": t("app.errors.no_project")}, status_code=404)
    if kind not in ORTHO_MEDIA_TYPES:
        return JSONResponse({"error": t("app.errors.invalid_ortho_kind", kind=kind)}, status_code=400)
    try:
        controller = DxfPageController(project_directory)
        level_id = None if level_key == "site" else level_key
        path = controller.ortho_file(level_id, kind)
    except ValueError as error:
        return JSONResponse({"error": str(error)}, status_code=400)
    except Exception as error:
        return JSONResponse({"error": str(error)}, status_code=500)
    if path is None or not path.is_file():
        return JSONResponse({"error": t("app.errors.no_ortho_artifact", kind=kind, level=level_key)}, status_code=404)
    return FileResponse(path, media_type=ORTHO_MEDIA_TYPES[kind], headers={"Cache-Control": "private, max-age=3600"})


def _bim_file(level_key: str, finder) -> tuple[Path | None, JSONResponse | None]:
    """Resolve a BIM artefact of the active project; returns (path, None) or (None, error response)."""
    project_directory = state.active_project_directory
    if project_directory is None:
        return None, JSONResponse({"error": t("app.errors.no_project")}, status_code=404)
    try:
        path = finder(project_directory, level_key)
    except ValueError as error:
        return None, JSONResponse({"error": str(error)}, status_code=400)
    if path is None or not path.is_file():
        return None, JSONResponse({"error": t("app.errors.no_bim_file", level=level_key)}, status_code=404)
    return path, None


@app.get("/api/bim/{level_key}/model.glb")
def bim_model(level_key: str):
    """Serve the IFC viewer GLB of a level; never generates."""
    path, error = _bim_file(level_key, bim_ui_data.glb_path)
    if path is None:
        return error
    return FileResponse(path, media_type="model/gltf-binary", headers={"Cache-Control": "no-cache"})


@app.get("/api/bim/{level_key}/props.json")
def bim_props(level_key: str):
    """Serve the IFC element properties (props.json) of a level."""
    path, error = _bim_file(level_key, bim_ui_data.props_path)
    if path is None:
        return error
    return FileResponse(path, media_type="application/json", headers={"Cache-Control": "no-cache"})


@app.get("/api/bim/{level_key}/ifc")
def bim_ifc(level_key: str):
    """Download the IFC file of a level as an attachment."""
    path, error = _bim_file(level_key, bim_ui_data.find_ifc)
    if path is None:
        return error
    return FileResponse(path, media_type="application/x-step", filename=path.name, headers={"Cache-Control": "no-cache"})


def _review_action(payload: dict[str, object], action):
    project_directory = state.active_project_directory
    if project_directory is None:
        return JSONResponse({"error": t("app.errors.no_project")}, status_code=404)
    if not isinstance(payload, dict):
        return JSONResponse({"error": t("app.errors.invalid_body")}, status_code=400)
    try:
        result = action(project_directory, payload)
    except ValueError as error:
        return JSONResponse({"error": str(error)}, status_code=400)
    return JSONResponse({"status": "ok", "result": result})


def _required_text(payload: dict[str, object], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(t("app.errors.field_required", field=key))
    return value


def create_application() -> None:
    """Register every page before the NiceGUI server starts."""
    app.add_static_files("/assets-static", ASSETS_DIRECTORY)
    ui.page("/")(home.render)
    ui.page("/projects")(projects.render)
    ui.page("/sparse")(sparse_cloud.render)
    ui.page("/control-points")(control_points.render)
    ui.page("/targets")(targets.render)
    ui.page("/dense")(dense_cloud.render)
    ui.page("/dxf")(dxf.render)
    ui.page("/settings")(settings.render)
