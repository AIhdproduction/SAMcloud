"""Sparse cloud and camera inspection page."""

from nicegui import ui

from samcloud.app.i18n import t, translate_or
from samcloud.app.pages.gcp_review import render_review_workspace
from samcloud.app.layout import render_page
from samcloud.app.pointcloud_viewer import render_pointcloud_viewer
from samcloud.app.state import state


def render() -> None:
    """Render the sparse cloud, camera, and control-point overview."""
    def content() -> None:
        project = state.active_project()
        if project is None:
            ui.label(t("sparse.no_project"))
            return
        ui.label(t("sparse.subtitle")).classes("sam-muted")
        with ui.column().classes("w-full gap-4"):
            with ui.card().classes("w-full"):
                render_pointcloud_viewer("sparse", title=t("sparse.viewer_title"), marker_endpoint="/api/control-points/markers")
            if (state.active_project_directory / "control_points" / "detections.json").is_file():
                render_review_workspace()
            with ui.card().classes("w-full"):
                ui.label(t("sparse.alignment_status")).classes("text-lg font-medium")
                alignment_status = str(project["workflow"]["alignment"])
                ui.label(translate_or(
                    f"workflow.status.{alignment_status.replace(' ', '_')}",
                    alignment_status.replace("_", " ").title(),
                ))
                ui.separator()
                ui.label(t("sparse.hint")).classes("sam-muted")

    render_page(t("sparse.title"), content, current_path="/sparse", wide=True)
