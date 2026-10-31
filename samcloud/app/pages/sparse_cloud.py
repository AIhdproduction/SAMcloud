"""Sparse cloud and camera inspection page."""

from nicegui import ui

from samcloud.app.layout import render_page
from samcloud.app.pointcloud_viewer import render_pointcloud_viewer
from samcloud.app.state import state


def render() -> None:
    """Render the sparse cloud, camera, and control-point overview."""
    def content() -> None:
        project = state.active_project()
        if project is None:
            ui.label("Open a project before viewing alignment results.")
            return
        ui.label("COLMAP sparse reconstruction").classes("sam-muted")
        with ui.column().classes("w-full gap-4"):
            with ui.card().classes("w-full"):
                render_pointcloud_viewer("sparse", title="Sparse point cloud")
            with ui.card().classes("w-full"):
                ui.label("Alignment status").classes("text-lg font-medium")
                ui.label(project["workflow"]["alignment"].replace("_", " ").title())
                ui.separator()
                ui.label("The view shows a browser preview of the reconstructed COLMAP points. The original model remains in products/sparse.").classes("sam-muted")

    render_page("Sparse Cloud", content, current_path="/sparse")
