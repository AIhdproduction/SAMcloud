"""Route registration for the NiceGUI application."""

from nicegui import app, ui
from starlette.responses import FileResponse, JSONResponse, Response

from samcloud.app.pages import control_points, dense_cloud, home, projects, settings, sparse_cloud
from samcloud.app.state import state
from samcloud.app.gcp_review import image_crop, marker_payload, resolve_project_image
from samcloud.core.paths import ASSETS_DIRECTORY
from samcloud.control_points.repository import (
    move_detection_observation,
    remove_detection_observation,
    set_control_point_role,
    set_detection_enabled,
    set_detection_observation_enabled,
    set_detection_role,
)
from samcloud.viewer.preview import ensure_preview
from samcloud.viewer.potree import ensure_potree, potree_status, start_potree_build
from samcloud.viewer.cameras import camera_payload, camera_thumbnail


@app.get("/api/pointcloud/{kind}")
def pointcloud_preview(kind: str):
    """Serve a compact, browser-friendly preview of the active project's cloud."""
    project_directory = state.active_project_directory
    if project_directory is None:
        return JSONResponse({"error": "No project is open"}, status_code=404)
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
        return JSONResponse({"error": "No project is open"}, status_code=404)
    try:
        return JSONResponse(camera_payload(project_directory))
    except (FileNotFoundError, OSError, ValueError) as error:
        return JSONResponse({"error": str(error)}, status_code=404)


@app.get("/api/camera-thumbnail/{camera_id}")
def sparse_camera_thumbnail(camera_id: str, size: str = "320"):
    """Serve a downscaled, ID-addressed source image for a sparse camera."""
    project_directory = state.active_project_directory
    if project_directory is None:
        return JSONResponse({"error": "No project is open"}, status_code=404)
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
        return JSONResponse({"error": "No project is open"}, status_code=404)
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
        return JSONResponse({"error": "No project is open"}, status_code=404)
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
        return JSONResponse({"error": "Invalid path"}, status_code=400)
    if not candidate.is_file():
        return JSONResponse({"error": "File not found"}, status_code=404)
    return FileResponse(candidate, headers={"Cache-Control": "no-store"})


@app.get("/api/control-point-image/{image_path:path}")
def control_point_image(image_path: str):
    """Serve only a file below the active project's dense image directory."""
    project_directory = state.active_project_directory
    if project_directory is None:
        return JSONResponse({"error": "No project is open"}, status_code=404)
    image_root = (project_directory / "products" / "dense" / "images").resolve()
    if not image_root.is_dir():
        try:
            return FileResponse(
                resolve_project_image(project_directory, image_path, "original"),
                headers={"Cache-Control": "no-store"},
            )
        except (FileNotFoundError, ValueError):
            return JSONResponse({"error": "Image not found"}, status_code=404)
    candidate = (image_root / image_path).resolve()
    try:
        candidate.relative_to(image_root)
    except ValueError:
        return JSONResponse({"error": "Invalid image path"}, status_code=400)
    if not candidate.is_file():
        return JSONResponse({"error": "Image not found"}, status_code=404)
    return FileResponse(candidate, headers={"Cache-Control": "no-store"})


@app.get("/api/control-points/markers")
def control_point_markers():
    """Serve reviewed markers, errors, and final-model coordinates for the sparse view."""
    project_directory = state.active_project_directory
    if project_directory is None:
        return JSONResponse({"error": "No project is open"}, status_code=404)
    return JSONResponse(marker_payload(project_directory))


@app.get("/api/control-points/crop")
def control_point_crop(image: str, u: str, v: str, size: str = "96", source: str = "original"):
    """Serve a small, path-safe original or undistorted marker crop."""
    project_directory = state.active_project_directory
    if project_directory is None:
        return JSONResponse({"error": "No project is open"}, status_code=404)
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
        return JSONResponse({"error": "No project is open"}, status_code=404)
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
        return JSONResponse({"error": "No project is open"}, status_code=404)
    try:
        state.processing.continue_after_control_points(project_directory)
    except (OSError, ValueError) as error:
        return JSONResponse({"error": str(error)}, status_code=409)
    return JSONResponse({"status": "started"}, status_code=202)


def _review_action(payload: dict[str, object], action):
    project_directory = state.active_project_directory
    if project_directory is None:
        return JSONResponse({"error": "No project is open"}, status_code=404)
    if not isinstance(payload, dict):
        return JSONResponse({"error": "Request body must be an object"}, status_code=400)
    try:
        result = action(project_directory, payload)
    except ValueError as error:
        return JSONResponse({"error": str(error)}, status_code=400)
    return JSONResponse({"status": "ok", "result": result})


def _required_text(payload: dict[str, object], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{key} is required")
    return value


def create_application() -> None:
    """Register every page before the NiceGUI server starts."""
    app.add_static_files("/assets-static", ASSETS_DIRECTORY)
    ui.page("/")(home.render)
    ui.page("/projects")(projects.render)
    ui.page("/sparse")(sparse_cloud.render)
    ui.page("/control-points")(control_points.render)
    ui.page("/dense")(dense_cloud.render)
    ui.page("/settings")(settings.render)
