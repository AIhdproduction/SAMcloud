"""Sparse cloud and camera inspection page."""

from nicegui import ui

from samcloud.app.pages.gcp_review import render_review_workspace
from samcloud.app.layout import render_page
from samcloud.app.pointcloud_viewer import render_pointcloud_viewer
from samcloud.app.state import state


def render() -> None:
    """Render the sparse cloud, camera, and control-point overview."""
    def content() -> None:
        project = state.active_project()
        if project is None:
            ui.label("Öffnen Sie zuerst ein Projekt, um die Ausrichtung anzuzeigen.")
            return
        ui.label("COLMAP Sparse-Rekonstruktion").classes("sam-muted")
        with ui.column().classes("w-full gap-4"):
            with ui.card().classes("w-full"):
                render_pointcloud_viewer("sparse", title="Sparse Punktwolke", marker_endpoint="/api/control-points/markers")
            if (state.active_project_directory / "control_points" / "detections.json").is_file():
                render_review_workspace()
            with ui.card().classes("w-full"):
                ui.label("Ausrichtungsstatus").classes("text-lg font-medium")
                ui.label(project["workflow"]["alignment"].replace("_", " ").title())
                ui.separator()
                ui.label("Die Ansicht zeigt eine Browser-Vorschau der rekonstruierten COLMAP-Punkte. Das Originalmodell bleibt unter products/sparse erhalten.").classes("sam-muted")

    render_page("Sparse Cloud", content, current_path="/sparse", wide=True)
