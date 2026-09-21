"""Project dashboard page."""

from pathlib import Path

from nicegui import ui

from samcloud.app.layout import render_page
from samcloud.app.state import state


_WORKFLOW = (
    ("Acquire imagery", None),
    ("Align cameras", "alignment"),
    ("Add control points", "control_points"),
    ("Build dense cloud", "dense_cloud"),
    ("Classify with SAM3", "classification"),
    ("Export LAS / PLY", "export"),
)


def render() -> None:
    """Render the active-project dashboard or a practical starting workspace."""
    project = state.active_project()
    title = "Ready for your next reconstruction?" if project is None else project["project"]["name"].replace("-", " ").title()

    def content() -> None:
        with ui.row().classes("w-full gap-5 items-stretch").style("align-items: stretch"):
            with ui.column().classes("flex-1 gap-0").style("min-width: min(100%, 520px)"):
                _render_project_workspace(project)
            with ui.element("aside").classes("sam-surface sam-workflow").style("width: min(100%, 310px)"):
                _render_workflow(project)
        _render_recent_projects()

    render_page(title, content, current_path="/")


def _render_project_workspace(project: dict | None) -> None:
    """Render the useful entry action for either an empty or active workspace."""
    with ui.element("div").classes("sam-dropzone column items-center justify-center text-center"):
        ui.icon("photo_library")
        if project is None:
            ui.label("Drop drone images or a 360° source here").classes("sam-dropzone-title")
            ui.label("Create a project first, then select the image directory or panorama source.").classes("sam-muted text-sm q-mt-sm")
            with ui.row().classes("q-mt-xl gap-3"):
                ui.button("Create project", icon="add", on_click=lambda: ui.navigate.to("/projects")).classes("sam-button-primary")
                ui.button("Open project", icon="folder_open", on_click=lambda: ui.navigate.to("/projects")).props("outline").classes("sam-button-secondary")
            return

        acquisition = project.get("acquisition", {})
        sources = acquisition.get("sources", project.get("paths", {}))
        capture_type = acquisition.get("type", "drone_gps")
        source = (
            sources.get("panorama_image_directory") or sources.get("panorama_video_path")
            if capture_type == "panorama_360"
            else sources.get("image_directory", "")
        )
        ui.icon("folder_open")
        ui.label("Project workspace is ready").classes("sam-dropzone-title")
        ui.label(source or "No image source selected yet").classes("sam-muted text-sm q-mt-sm").style("max-width: 80%; overflow-wrap: anywhere")
        with ui.row().classes("q-mt-xl gap-3"):
            ui.button("Review project settings", icon="tune", on_click=lambda: ui.navigate.to("/settings")).classes("sam-button-primary")
            ui.button("Open sparse cloud", icon="hub", on_click=lambda: ui.navigate.to("/sparse")).props("outline").classes("sam-button-secondary")


def _render_workflow(project: dict | None) -> None:
    ui.label("Project workflow").classes("sam-workflow-title")
    statuses = project.get("workflow", {}) if project else {}
    for index, (label, key) in enumerate(_WORKFLOW, start=1):
        status = "ready" if key is None and project is not None else statuses.get(key, "not started")
        step_class = "sam-workflow-step"
        if index == 1 and project is None:
            step_class += " is-active"
        elif status == "completed":
            step_class += " is-complete"
        elif status in {"running", "ready"}:
            step_class += " is-active"
        with ui.element("div").classes(step_class):
            ui.label(str(index)).classes("sam-step-number")
            ui.label(label).classes("sam-step-label")
            ui.label(status.replace("_", " ")).classes("sam-step-status")


def _render_recent_projects() -> None:
    projects = state.projects.discover_projects()
    with ui.element("section").classes("sam-surface sam-recent"):
        with ui.row().classes("sam-recent-head items-center justify-between w-full"):
            ui.label("Recent projects").classes("sam-section-title")
            ui.button("View all projects", icon="arrow_forward", on_click=lambda: ui.navigate.to("/projects")).props("flat no-caps").classes("text-cyan-4")
        if not projects:
            ui.label("No saved projects yet. Create your first project to begin the workflow.").classes("sam-empty-row")
            return
        for project_directory in projects[:5]:
            project = state.projects.load(project_directory)
            with ui.row().classes("sam-project-row w-full items-center no-wrap"):
                ui.icon("folder").classes("text-cyan-4 q-mr-md")
                with ui.column().classes("gap-0 flex-1").style("min-width: 0"):
                    ui.label(project["project"]["name"]).classes("sam-project-name")
                    ui.label(str(project_directory)).classes("sam-project-path")
                ui.button("Open", on_click=lambda path=project_directory: _open_project(path)).props("flat no-caps").classes("text-cyan-4")


def _open_project(project_directory: Path) -> None:
    state.open_project(project_directory)
    ui.navigate.to("/")
