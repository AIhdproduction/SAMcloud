"""Sparse cloud and camera inspection page."""

from nicegui import ui

from samcloud.app.components.pointcloud_viewer import render_pointcloud_viewer
from samcloud.app.layout import render_page
from samcloud.app.state import state


def render() -> None:
    """Render the sparse point cloud preview and alignment summary."""
    def content() -> None:
        project = state.active_project()
        if project is None:
            ui.label("Open a project before viewing alignment results.")
            return
        with ui.card().classes("w-full p-0 overflow-hidden"):
            render_pointcloud_viewer("sparse", title="Sparse point cloud")
        with ui.card().classes("w-full mt-4"):
            ui.label("Alignment status").classes("text-lg font-medium")
            ui.label(project["workflow"]["alignment"].replace("_", " ").title())
            ui.separator()
            ui.label("Control point observations will be listed here.").classes("sam-muted")

    render_page("Sparse Cloud", content, current_path="/sparse")
