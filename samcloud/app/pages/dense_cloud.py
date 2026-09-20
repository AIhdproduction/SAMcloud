"""Dense cloud inspection and semantic class controls."""

import json

from nicegui import ui

from samcloud.app.layout import render_page
from samcloud.app.state import state
from samcloud.core.paths import CONFIG_DIRECTORY


LAS_CODES = (0, 2, 11, 66, 67, 3, 5, 6, 68, 69, 70, 71, 72, 14, 73, 74, 9)


def render() -> None:
    """Render RGB/classification mode and per-class visibility controls."""
    def content() -> None:
        project = state.active_project()
        if project is None:
            ui.label("Open a project before viewing dense point cloud results.")
            return
        classes = _load_classes()
        with ui.row().classes("w-full gap-4"):
            with ui.card().classes("flex-1 min-h-96"):
                ui.label("Dense point cloud viewer").classes("text-lg font-medium")
                ui.label("Potree will render the project LAS or converted viewer data here.")
                ui.radio(["RGB", "Classification"], value="RGB").props("inline")
            with ui.card().classes("w-96"):
                ui.label("Class visibility").classes("text-lg font-medium")
                ui.checkbox("Unknown / noise", value=True)
                for name, las_code in zip(classes, LAS_CODES[1:]):
                    ui.checkbox(f"{las_code}: {name.split(',')[0].title()}", value=True)
                ui.button("Show all")
                ui.button("Hide all").props("outline")

    render_page("Dense Cloud", content)


def _load_classes() -> list[str]:
    data = json.loads((CONFIG_DIRECTORY / "classes.json").read_text(encoding="utf-8"))
    return data["aussen"]
