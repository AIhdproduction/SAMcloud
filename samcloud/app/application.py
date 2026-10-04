"""Route registration for the NiceGUI application."""

import asyncio

from fastapi.responses import FileResponse, JSONResponse
from nicegui import app, ui

from samcloud.app.pages import control_points, dense_cloud, home, projects, settings, sparse_cloud
from samcloud.app.state import state
from samcloud.viewer.preview import ensure_preview


async def pointcloud_preview(kind: str):
    """Serve the compact point cloud preview for the active project only."""
    if state.active_project_directory is None:
        return JSONResponse({"error": "No project is open"}, status_code=404)
    try:
        preview, point_count = await asyncio.to_thread(
            ensure_preview, state.active_project_directory, kind
        )
    except (FileNotFoundError, OSError, ValueError) as error:
        return JSONResponse({"error": str(error)}, status_code=404)
    return FileResponse(
        preview,
        media_type="application/octet-stream",
        headers={"Cache-Control": "no-store", "X-SAMcloud-Point-Count": str(point_count)},
    )


def create_application() -> None:
    """Register every page before the NiceGUI server starts."""
    ui.page("/")(home.render)
    ui.page("/projects")(projects.render)
    ui.page("/sparse")(sparse_cloud.render)
    ui.page("/control-points")(control_points.render)
    ui.page("/dense")(dense_cloud.render)
    ui.page("/settings")(settings.render)

    app.get("/api/pointcloud/{kind}")(pointcloud_preview)
