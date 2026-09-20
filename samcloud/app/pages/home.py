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
        acquisition = project.get("acquisition", {})
        capture_labels = {
            "drone_gps": "Drone with GPS / RTK",
            "camera": "Standard camera without GPS",
            "panorama_360": "360 degree camera / Insta360",
        }
        capture_type = acquisition.get("type", "drone_gps")
        with ui.card().classes("w-full"):
            ui.label("Capture summary").classes("text-lg font-medium")
            ui.label(f"Capture type: {capture_labels.get(capture_type, 'Legacy project')}")
            sources = acquisition.get("sources", project.get("paths", {}))
            if capture_type == "panorama_360":
                if sources.get("panorama_video_path"):
                    ui.label(f"360 degree video: {sources['panorama_video_path']}").classes("text-sm")
                if sources.get("panorama_image_directory"):
                    ui.label(f"360 degree photos: {sources['panorama_image_directory']}").classes("text-sm")
                prepared = acquisition.get("panorama", {}).get("prepared", {})
                ui.label(
                    f"Extracted video frames: {prepared.get('video_frame_count', 0)} | "
                    f"Virtual camera images: {prepared.get('cubemap_image_count', 0)}"
                ).classes("text-sm")
            else:
                ui.label(f"Image directory: {sources.get('image_directory', '')}").classes("text-sm")
            ui.label(
                f"Georeferencing: {acquisition.get('georeferencing', {}).get('mode', 'gps').upper()}"
            ).classes("text-sm")
        with ui.row().classes("w-full gap-4"):
            for stage, status in project["workflow"].items():
                with ui.card().classes("w-48"):
                    ui.label(stage.replace("_", " ").title()).classes("font-medium")
                    ui.label(status.replace("_", " ").title())
        ui.label("Use the navigation to align images, define control points, inspect point clouds, and export LAS.")

    render_page("Project dashboard", content)
