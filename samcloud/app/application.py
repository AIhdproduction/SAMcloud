"""Route registration for the NiceGUI application."""

from nicegui import app, ui
from starlette.responses import FileResponse, JSONResponse

from samcloud.app.pages import control_points, dense_cloud, home, projects, settings, sparse_cloud
from samcloud.app.state import state
from samcloud.core.paths import ASSETS_DIRECTORY
from samcloud.viewer.preview import ensure_preview


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


@app.get("/api/control-point-image/{image_path:path}")
def control_point_image(image_path: str):
    """Serve only a file below the active project's dense image directory."""
    project_directory = state.active_project_directory
    if project_directory is None:
        return JSONResponse({"error": "No project is open"}, status_code=404)
    image_root = (project_directory / "products" / "dense" / "images").resolve()
    candidate = (image_root / image_path).resolve()
    try:
        candidate.relative_to(image_root)
    except ValueError:
        return JSONResponse({"error": "Invalid image path"}, status_code=400)
    if not candidate.is_file():
        return JSONResponse({"error": "Image not found"}, status_code=404)
    return FileResponse(candidate, headers={"Cache-Control": "no-store"})


def create_application() -> None:
    """Register every page before the NiceGUI server starts."""
    app.add_static_files("/assets-static", ASSETS_DIRECTORY)
    ui.page("/")(home.render)
    ui.page("/projects")(projects.render)
    ui.page("/sparse")(sparse_cloud.render)
    ui.page("/control-points")(control_points.render)
    ui.page("/dense")(dense_cloud.render)
    ui.page("/settings")(settings.render)
