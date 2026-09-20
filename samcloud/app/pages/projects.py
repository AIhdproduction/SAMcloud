"""Project creation and reopening page."""

from pathlib import Path

from nicegui import ui

from samcloud.app.layout import render_page
from samcloud.app.state import state
from samcloud.core.configuration import load_global_config


def render() -> None:
    """Render the persistent project manager."""
    def content() -> None:
        panorama_defaults = load_global_config("defaults.yml").get("acquisition", {}).get("panorama", {})
        with ui.card().classes("w-full"):
            ui.label("Create project").classes("text-lg font-medium")
            name = ui.input("Project name", placeholder="example-site")
            capture_type = ui.select(
                {
                    "drone_gps": "Drone with GPS / RTK",
                    "camera": "Standard camera without GPS",
                    "panorama_360": "360 degree camera / Insta360",
                },
                label="Capture type",
                value=None,
            )
            class_set = ui.select(
                {"outdoor": "Outdoor", "indoor": "Indoor"},
                label="Semantic class set",
                value=None,
            )

            with ui.column().classes("w-full gap-3").bind_visibility_from(
                capture_type, "value", lambda value: value in {"drone_gps", "camera"}
            ):
                image_directory = ui.input(
                    "Image directory", placeholder="D:\\data\\camera-images"
                ).classes("w-full")
                rtk_enabled = ui.switch("Images use RTK GPS", value=False).bind_visibility_from(
                    capture_type, "value", lambda value: value == "drone_gps"
                )

            with ui.column().classes("w-full gap-3").bind_visibility_from(
                capture_type, "value", lambda value: value == "panorama_360"
            ):
                ui.label("Use an exported 2:1 equirectangular MP4 video, a JPG panorama directory, or both.").classes("text-sm")
                panorama_video_path = ui.input(
                    "360 degree MP4 video (optional)", placeholder="D:\\data\\tour.mp4"
                ).classes("w-full")
                panorama_image_directory = ui.input(
                    "360 degree JPG panorama directory (optional)",
                    placeholder="D:\\data\\insta360-photos",
                ).classes("w-full")
                frame_interval_seconds = ui.number(
                    "Video frame interval in seconds",
                    value=panorama_defaults.get("frame_interval_seconds", 1.0),
                    min=0.1,
                    step=0.1,
                ).classes("w-full")

            def create_project() -> None:
                try:
                    if capture_type.value is None:
                        raise ValueError("Select a capture type")
                    if class_set.value is None:
                        raise ValueError("Select the indoor or outdoor semantic class set")
                    project_directory = state.projects.create_project(
                        name.value,
                        image_directory.value or "",
                        capture_type=capture_type.value,
                        class_set=class_set.value,
                        panorama_image_directory=panorama_image_directory.value or "",
                        panorama_video_path=panorama_video_path.value or "",
                        frame_interval_seconds=float(frame_interval_seconds.value or 0),
                        rtk_enabled=rtk_enabled.value,
                    )
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
                        acquisition = project.get("acquisition", {})
                        capture_label = {
                            "drone_gps": "Drone with GPS / RTK",
                            "camera": "Standard camera without GPS",
                            "panorama_360": "360 degree camera / Insta360",
                        }.get(acquisition.get("type"), "Legacy project")
                        ui.label(capture_label).classes("text-xs text-slate-500")
                    ui.button("Open", on_click=lambda path=project_directory: _open_project(path))

    render_page("Projects", content)


def _open_project(project_directory: Path) -> None:
    state.open_project(project_directory)
    ui.navigate.to("/")
