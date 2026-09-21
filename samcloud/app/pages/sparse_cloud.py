"""Sparse cloud and camera inspection page."""

from nicegui import ui

from samcloud.app.layout import render_page
from samcloud.app.state import state


def render() -> None:
    """Render the sparse cloud, camera, and control-point overview."""
    def content() -> None:
        project = state.active_project()
        if project is None:
            ui.label("Open a project before viewing alignment results.")
            return
        ui.label("This page will show reconstructed cameras, sparse points, and control point rays after alignment.")
        with ui.row().classes("w-full gap-4"):
            with ui.card().classes("flex-1 min-h-80"):
                ui.label("Sparse cloud viewer").classes("text-lg font-medium")
                ui.label("The viewer data will be generated from the COLMAP sparse model.")
            with ui.card().classes("w-80"):
                ui.label("Alignment status").classes("text-lg font-medium")
                ui.label(project["workflow"]["alignment"].replace("_", " ").title())
                ui.separator()
                ui.label("Control point observations will be listed here.")

    render_page("Sparse Cloud", content, current_path="/sparse")
