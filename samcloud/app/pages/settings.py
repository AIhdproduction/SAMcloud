"""Project coordinate and processing settings page."""

from nicegui import ui

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
            coordinates["input_crs"] = input_crs.value
            coordinates["working_crs"] = working_crs.value
            coordinates["export_crs"] = export_crs.value
            settings["workflow"]["use_gpu"] = use_gpu.value
            settings["workflow"]["control_points_enabled"] = control_points_enabled.value
            classification["class_set"] = class_set.value
            state.projects.save(project_directory, project)
            ui.notify("Project settings saved")

        ui.button("Save settings", on_click=save_settings)

    render_page("Settings", content)
