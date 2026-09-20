"""Project coordinate and processing settings page."""

from nicegui import ui

from samcloud.acquisition import (
    CaptureValidationError,
    prepare_project_panorama,
    validate_capture_configuration,
)
from samcloud.app.layout import render_page
from samcloud.app.state import state


def render() -> None:
    """Render editable coordinate system and workflow settings."""
    def content() -> None:
        project_directory = state.active_project_directory
        project = state.active_project()
        if project_directory is None or project is None:
            ui.label("Open a project before editing settings.")
            return

        settings = project["settings"]
        coordinates = settings["coordinate_systems"]
        classification = settings.setdefault("classification", {"class_set": "outdoor"})
        acquisition = project.setdefault(
            "acquisition",
            {
                "type": "drone_gps",
                "locked": False,
                "sources": {"image_directory": project["paths"].get("image_directory", "")},
                "panorama": {"frame_interval_seconds": 1.0},
                "georeferencing": {"mode": "gps", "rtk_enabled": False},
            },
        )
        sources = acquisition.setdefault("sources", {})
        panorama = acquisition.setdefault("panorama", {})
        georeferencing = acquisition.setdefault("georeferencing", {})
        capture_type = acquisition.get("type", "drone_gps")
        acquisition_locked = bool(acquisition.get("locked", False))
        capture_label = {
            "drone_gps": "Drone with GPS / RTK",
            "camera": "Standard camera without GPS",
            "panorama_360": "360 degree camera / Insta360",
        }.get(capture_type, "Unknown")

        with ui.card().classes("w-full"):
            ui.label("Capture sources").classes("text-lg font-medium")
            ui.label(f"Capture type: {capture_label}")
            if acquisition_locked:
                ui.label("Capture settings are locked because processing has started.").classes("text-sm")
            if capture_type in {"drone_gps", "camera"}:
                image_directory = ui.input(
                    "Image directory", value=sources.get("image_directory", "")
                ).classes("w-full")
                image_directory.set_enabled(not acquisition_locked)
                rtk_enabled = ui.switch(
                    "Images use RTK GPS", value=bool(georeferencing.get("rtk_enabled", False))
                )
                rtk_enabled.set_visibility(capture_type == "drone_gps")
                rtk_enabled.set_enabled(not acquisition_locked)
                panorama_video_path = None
                panorama_image_directory = None
                frame_interval_seconds = None
            else:
                panorama_video_path = ui.input(
                    "360 degree MP4 video", value=sources.get("panorama_video_path", "")
                ).classes("w-full")
                panorama_image_directory = ui.input(
                    "360 degree JPG panorama directory",
                    value=sources.get("panorama_image_directory", ""),
                ).classes("w-full")
                frame_interval_seconds = ui.number(
                    "Video frame interval in seconds",
                    value=panorama.get("frame_interval_seconds", 1.0),
                    min=0.1,
                    step=0.1,
                ).classes("w-full")
                for element in (panorama_video_path, panorama_image_directory, frame_interval_seconds):
                    element.set_enabled(not acquisition_locked)
                image_directory = None
                rtk_enabled = None
        with ui.card().classes("w-full"):
            ui.label("Coordinate systems").classes("text-lg font-medium")
            input_crs = ui.input("Image/GNSS CRS", value=coordinates["input_crs"])
            working_crs = ui.input("Working CRS", value=coordinates["working_crs"])
            export_crs = ui.input("Export CRS", value=coordinates["export_crs"])
            ui.label("Use LOCAL, an EPSG code such as EPSG:2056, or another CRS accepted by PROJ.").classes("text-sm")
        with ui.card().classes("w-full"):
            ui.label("Processing settings").classes("text-lg font-medium")
            use_gpu = ui.switch("Use GPU", value=settings["workflow"]["use_gpu"])
            control_points_enabled = ui.switch("Enable control points", value=settings["workflow"]["control_points_enabled"])
            class_set = ui.select(
                {"outdoor": "Outdoor", "indoor": "Indoor"},
                value=classification["class_set"],
                label="Semantic class set",
            )

        def save_settings() -> None:
            try:
                if not acquisition_locked:
                    if capture_type in {"drone_gps", "camera"}:
                        sources["image_directory"] = image_directory.value or ""
                        project["paths"]["image_directory"] = sources["image_directory"]
                        georeferencing["rtk_enabled"] = bool(rtk_enabled.value)
                    else:
                        sources["panorama_video_path"] = panorama_video_path.value or ""
                        sources["panorama_image_directory"] = panorama_image_directory.value or ""
                        panorama["frame_interval_seconds"] = float(frame_interval_seconds.value or 0)
                    validate_capture_configuration(
                        capture_type,
                        image_directory=sources.get("image_directory", ""),
                        panorama_image_directory=sources.get("panorama_image_directory", ""),
                        panorama_video_path=sources.get("panorama_video_path", ""),
                        frame_interval_seconds=float(panorama.get("frame_interval_seconds", 1.0)),
                    )
                coordinates["input_crs"] = input_crs.value
                coordinates["working_crs"] = working_crs.value
                coordinates["export_crs"] = export_crs.value
                settings["workflow"]["use_gpu"] = use_gpu.value
                settings["workflow"]["control_points_enabled"] = control_points_enabled.value
                classification["class_set"] = class_set.value
                state.projects.save(project_directory, project)
                ui.notify("Project settings saved")
            except (CaptureValidationError, ValueError) as error:
                ui.notify(str(error), type="negative")

        ui.button("Save settings", on_click=save_settings)

        if capture_type == "panorama_360" and not acquisition_locked:
            def prepare_inputs() -> None:
                try:
                    sources["panorama_video_path"] = panorama_video_path.value or ""
                    sources["panorama_image_directory"] = panorama_image_directory.value or ""
                    panorama["frame_interval_seconds"] = float(frame_interval_seconds.value or 0)
                    validate_capture_configuration(
                        capture_type,
                        panorama_image_directory=sources["panorama_image_directory"],
                        panorama_video_path=sources["panorama_video_path"],
                        frame_interval_seconds=panorama["frame_interval_seconds"],
                    )
                    prepared = prepare_project_panorama(project_directory, project)
                    state.projects.save(project_directory, project)
                    ui.notify(
                        f"Prepared {prepared.panorama_count} panoramas and "
                        f"{prepared.cubemap_image_count} virtual camera images"
                    )
                    ui.navigate.to("/")
                except (CaptureValidationError, ValueError) as error:
                    ui.notify(str(error), type="negative")

            ui.button("Prepare 360 degree inputs", on_click=prepare_inputs).props("outline")

    render_page("Settings", content)
