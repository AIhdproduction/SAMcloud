"""Route registration for the NiceGUI application."""

from nicegui import ui

from samcloud.app.pages import control_points, dense_cloud, home, projects, settings, sparse_cloud


def create_application() -> None:
    """Register every page before the NiceGUI server starts."""
    ui.page("/")(home.render)
    ui.page("/projects")(projects.render)
    ui.page("/sparse")(sparse_cloud.render)
    ui.page("/control-points")(control_points.render)
    ui.page("/dense")(dense_cloud.render)
    ui.page("/settings")(settings.render)
