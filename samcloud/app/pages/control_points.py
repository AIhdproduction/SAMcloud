"""Ground-control-point page structure."""

from pathlib import Path

from nicegui import ui

from samcloud.app.layout import render_page
from samcloud.app.state import state
from samcloud.control_points.repository import load_observations


def render() -> None:
    """Render the future image-marking workspace and saved observation status."""
    def content() -> None:
        project_directory = state.active_project_directory
        if project_directory is None:
            ui.label("Open a project before managing control points.")
            return
        observations = load_observations(project_directory / "control_points" / "observations.json")
        ui.label("Import a CSV with name, x, y, z and mark every control point in at least two images.")
        with ui.row().classes("w-full gap-4"):
            with ui.card().classes("flex-1 min-h-96"):
                ui.label("Image marker workspace").classes("text-lg font-medium")
                ui.label("Image selection, zooming, and marker placement will be implemented here.")
            with ui.card().classes("w-80"):
                ui.label("Saved observations").classes("text-lg font-medium")
                ui.label(f"Control points with image observations: {len(observations)}")
                ui.button("Open project settings", on_click=lambda: ui.navigate.to("/settings"))

    render_page("Control Points", content)
