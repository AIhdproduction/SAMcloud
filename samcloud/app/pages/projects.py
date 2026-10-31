"""Project creation and reopening page."""

from pathlib import Path
from tkinter import TclError, Tk, filedialog

from nicegui import ui

from samcloud.app.layout import render_page
from samcloud.app.state import state
from samcloud.core.configuration import load_global_config
from samcloud.core.project_deletion import delete_project


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
                value="drone_gps",
            )
            class_set = ui.select(
                {"outdoor": "Outdoor", "indoor": "Indoor"},
                label="Semantic class set",
                value="outdoor",
            )
            control_points_mode = ui.select(
                {
                    "none": "No control points - continue automatically",
                    "after_alignment": "Use control points after alignment",
                },
                label="Ground control points",
                value="none",
            )

            with ui.column().classes("w-full gap-3").bind_visibility_from(
                capture_type, "value", lambda value: value in {"drone_gps", "camera"}
            ):
                image_directory = ui.input(
                    "Image directory", placeholder="D:\\data\\camera-images"
                ).classes("w-full")
                rtk_enabled = ui.switch("Images use RTK GPS", value=True)

                def choose_image_directory() -> None:
                    """Open the native Windows folder dialog for the image source."""
                    dialog_root = None
                    try:
                        dialog_root = Tk()
                        dialog_root.withdraw()
                        dialog_root.attributes("-topmost", True)
                        current_directory = Path(image_directory.value or "")
                        selected_directory = filedialog.askdirectory(
                            initialdir=current_directory if current_directory.is_dir() else None,
                            parent=dialog_root,
                            title="Select image directory",
                            mustexist=True,
                        )
                    except TclError as error:
                        ui.notify(f"Could not open the folder dialog: {error}", type="negative")
                        return
                    finally:
                        if dialog_root is not None:
                            dialog_root.destroy()
                    if selected_directory:
                        image_directory.value = selected_directory

                with image_directory.add_slot("append"):
                    ui.button(
                        icon="folder_open",
                        on_click=choose_image_directory,
                    ).props("flat round dense aria-label='Select image directory'")

                def update_rtk_option() -> None:
                    is_rtk_capture = capture_type.value == "drone_gps"
                    rtk_enabled.set_visibility(is_rtk_capture)
                    rtk_enabled.value = is_rtk_capture

                capture_type.on_value_change(lambda _: update_rtk_option())
                update_rtk_option()

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
                        control_points_mode=control_points_mode.value,
                    )
                    state.open_project(project_directory)
                    ui.notify(f"Project created: {project_directory.name}")
                    ui.navigate.to("/")
                except (FileExistsError, ValueError) as error:
                    ui.notify(str(error), type="negative")

            ui.button("Create project", on_click=create_project)

        ui.label("Existing projects").classes("text-lg font-medium")

        @ui.refreshable
        def render_existing_projects() -> None:
            projects = state.projects.discover_projects()
            if not projects:
                ui.label("No saved projects found.")
                return
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
                        with ui.row().classes("items-center gap-2"):
                            ui.button(
                                "Open",
                                on_click=lambda path=project_directory: _open_project(path),
                            )
                            ui.button(
                                "Delete",
                                icon="delete",
                                on_click=lambda path=project_directory: _confirm_delete(
                                    path, render_existing_projects.refresh
                                ),
                            ).props("outline color=negative")

        render_existing_projects()

    render_page("Projects", content, current_path="/projects")


def _open_project(project_directory: Path) -> None:
    state.open_project(project_directory)
    ui.navigate.to("/")


def _project_is_processing(project_directory: Path) -> bool:
    """Protect projects with active persisted or in-memory processing jobs."""
    if state.processing.is_running(project_directory):
        return True
    project = state.projects.load(project_directory)
    return any(
        status in {"running", "queued"}
        for status in project.get("workflow", {}).values()
        if isinstance(status, str)
    )


def _confirm_delete(project_directory: Path, refresh_projects) -> None:
    """Ask for confirmation before deleting a project and all its outputs."""
    if _project_is_processing(project_directory):
        ui.notify("This project is still processing and cannot be deleted.", type="warning")
        return

    with ui.dialog() as dialog, ui.card().classes("w-full"):
        ui.label("Delete project?").classes("text-lg font-medium")
        ui.label(
            f"This permanently removes {project_directory.name} and all generated files."
        ).classes("text-sm")
        with ui.row().classes("justify-end w-full gap-2 q-mt-md"):
            ui.button("Cancel", on_click=dialog.close).props("flat")

            def delete_confirmed() -> None:
                try:
                    delete_project(state.projects.projects_directory, project_directory)
                    if state.active_project_directory == project_directory.resolve():
                        state.active_project_directory = None
                    dialog.close()
                    refresh_projects()
                    ui.notify(f"Project deleted: {project_directory.name}")
                except (OSError, ValueError) as error:
                    ui.notify(str(error), type="negative")

            ui.button("Delete permanently", icon="delete", on_click=delete_confirmed).props(
                "color=negative"
            )
    dialog.open()
