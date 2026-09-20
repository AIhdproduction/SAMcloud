"""Project creation and reopening page."""

from pathlib import Path

from nicegui import ui

from samcloud.app.layout import render_page
from samcloud.app.state import state


def render() -> None:
    """Render the persistent project manager."""
    def content() -> None:
        with ui.card().classes("w-full"):
            ui.label("Create project").classes("text-lg font-medium")
            name = ui.input("Project name", placeholder="example-site")
            image_directory = ui.input("Image directory", placeholder="D:\\data\\drone-images")

            def create_project() -> None:
                try:
                    project_directory = state.projects.create_project(name.value, image_directory.value or "")
                    state.open_project(project_directory)
                    ui.notify(f"Project created: {project_directory.name}")
                    ui.navigate.to("/")
                except (FileExistsError, ValueError) as error:
                    ui.notify(str(error), type="negative")

            ui.button("Create project", on_click=create_project)

        ui.label("Existing projects").classes("text-lg font-medium")
        projects = state.projects.discover_projects()
        if not projects:
            ui.label("No saved projects found.")
        for project_directory in projects:
            project = state.projects.load(project_directory)
            with ui.card().classes("w-full"):
                with ui.row().classes("items-center justify-between w-full"):
                    with ui.column().classes("gap-0"):
                        ui.label(project["project"]["name"]).classes("font-medium")
                        ui.label(str(project_directory)).classes("text-xs text-slate-500")
                    ui.button("Open", on_click=lambda path=project_directory: _open_project(path))

    render_page("Projects", content)


def _open_project(project_directory: Path) -> None:
    state.open_project(project_directory)
    ui.navigate.to("/")
