"""Landing page."""

from nicegui import ui

from samcloud.app.layout import render_page
from samcloud.app.state import state


def render() -> None:
    """Render the project dashboard."""
    def content() -> None:
        project = state.active_project()
        if project is None:
            ui.label("Create or open a project to begin.").classes("text-lg")
            ui.button("Manage projects", on_click=lambda: ui.navigate.to("/projects"))
            return

        project_name = project["project"]["name"]
        ui.label(f"Open project: {project_name}").classes("text-lg")
        with ui.row().classes("w-full gap-4"):
            for stage, status in project["workflow"].items():
                with ui.card().classes("w-48"):
                    ui.label(stage.replace("_", " ").title()).classes("font-medium")
                    ui.label(status.replace("_", " ").title())
        ui.label("Use the navigation to align images, define control points, inspect point clouds, and export LAS.")

    render_page("Project dashboard", content)
