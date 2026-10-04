"""Dense cloud inspection and semantic class controls."""

from __future__ import annotations

import json
import re

from nicegui import ui

from samcloud.app.components.pointcloud_viewer import render_pointcloud_viewer
from samcloud.app.layout import render_page
from samcloud.app.state import state
from samcloud.core.paths import CONFIG_DIRECTORY


LAS_CODES = {
    "outdoor": (0, 2, 11, 66, 67, 3, 5, 6, 68, 69, 70, 71, 72, 14, 73, 74, 9),
    "indoor": (0, 2, 75, 6, 76, 77, 78, 79, 80, 81, 82),
}
_DEFAULT_COLORS = (
    (189, 189, 189), (166, 124, 82), (77, 77, 77), (184, 184, 184), (255, 210, 63),
    (126, 200, 80), (34, 139, 34), (217, 83, 79), (0, 162, 232), (255, 140, 0),
    (139, 90, 43), (255, 0, 255), (138, 43, 226), (0, 206, 209), (255, 215, 0),
    (255, 99, 71), (30, 144, 255),
)
_HEX_COLOR = re.compile(r"^#[0-9A-Fa-f]{6}$")


def render() -> None:
    """Render the dense cloud preview and persistent class controls."""
    def content() -> None:
        project = state.active_project()
        if project is None:
            ui.label("Open a project before viewing dense point cloud results.")
            return
        class_set = project["settings"].get("classification", {}).get("class_set", "outdoor")
        classes = _load_classes(class_set)
        labels = ["Unknown / noise"] + [
            f"{code}: {name.split(',')[0].title()}"
            for name, code in zip(classes, LAS_CODES[class_set][1:])
        ]
        colors = _project_colors(project, len(labels))
        visible = [True] * len(labels)

        with ui.card().classes("w-full p-0 overflow-hidden"):
            render_pointcloud_viewer(
                "dense", title="Dense point cloud", class_colors=colors, class_labels=labels
            )
        with ui.card().classes("w-full mt-4"):
            ui.label("Classification visibility and colors").classes("text-lg font-medium")
            ui.label("Changes are saved in this project and applied to the preview immediately.").classes("sam-muted")

            def apply_colors() -> None:
                ui.run_javascript(
                    "window.samcloudPointCloudViewers?.['sam-cloud-dense']?.setClassColors("
                    + json.dumps(colors) + ");"
                )

            def apply_visibility() -> None:
                ui.run_javascript(
                    "window.samcloudPointCloudViewers?.['sam-cloud-dense']?.setClassVisibility("
                    + json.dumps(visible) + ");"
                )

            def save_colors() -> None:
                settings = project.setdefault("settings", {}).setdefault("classification", {})
                settings["class_colors"] = {
                    str(index): _rgb_to_hex(color) for index, color in enumerate(colors)
                }
                state.projects.save(state.active_project_directory, project)

            checkboxes: list = []
            color_inputs: list = []
            with ui.row().classes("gap-2 mt-3"):
                def show_all() -> None:
                    visible[:] = [True] * len(visible)
                    for checkbox in checkboxes:
                        checkbox.value = True
                    apply_visibility()

                def hide_all() -> None:
                    visible[:] = [False] * len(visible)
                    for checkbox in checkboxes:
                        checkbox.value = False
                    apply_visibility()

                def restore_defaults() -> None:
                    colors[:] = [list(_DEFAULT_COLORS[index % len(_DEFAULT_COLORS)]) for index in range(len(colors))]
                    for index, color_input in enumerate(color_inputs):
                        color_input.value = _rgb_to_hex(colors[index])
                    save_colors()
                    apply_colors()

                ui.button("Show all", on_click=show_all).props("outline")
                ui.button("Hide all", on_click=hide_all).props("outline")
                ui.button("Restore default colors", on_click=restore_defaults).props("outline")

            for index, label in enumerate(labels):
                with ui.row().classes("w-full items-center gap-3"):
                    checkbox = ui.checkbox(label, value=True)
                    checkbox.on_value_change(
                        lambda event, selected=index: _set_visibility(
                            visible, selected, bool(event.value), apply_visibility
                        )
                    )
                    checkboxes.append(checkbox)
                    ui.button(
                        "Isolate",
                        on_click=lambda selected=index: _isolate(
                            visible, checkboxes, selected, apply_visibility
                        ),
                    ).props("flat dense")
                    color_input = ui.color_input(value=_rgb_to_hex(colors[index]), preview=True)
                    color_input.on_value_change(
                        lambda event, selected=index: _set_color(
                            colors, selected, event.value, save_colors, apply_colors
                        )
                    )
                    color_inputs.append(color_input)

    render_page("Dense Cloud", content, current_path="/dense")


def _load_classes(class_set: str) -> list[str]:
    data = json.loads((CONFIG_DIRECTORY / "classes.json").read_text(encoding="utf-8"))
    return data[class_set]


def _project_colors(project: dict, count: int) -> list[list[int]]:
    configured = project.get("settings", {}).get("classification", {}).get("class_colors", {})
    return [
        _hex_to_rgb(configured.get(str(index)))
        if isinstance(configured.get(str(index)), str) and _HEX_COLOR.fullmatch(configured[str(index)])
        else list(_DEFAULT_COLORS[index % len(_DEFAULT_COLORS)])
        for index in range(count)
    ]


def _hex_to_rgb(value: str) -> list[int]:
    return [int(value[index:index + 2], 16) for index in (1, 3, 5)]


def _rgb_to_hex(value: list[int] | tuple[int, int, int]) -> str:
    return "#{:02X}{:02X}{:02X}".format(*value)


def _set_visibility(values: list[bool], index: int, selected: bool, callback) -> None:
    values[index] = selected
    callback()


def _isolate(values: list[bool], checkboxes: list, selected: int, callback) -> None:
    values[:] = [index == selected for index in range(len(values))]
    for index, checkbox in enumerate(checkboxes):
        checkbox.value = index == selected
    callback()


def _set_color(values: list[list[int]], index: int, value: str | None, save_callback, apply_callback) -> None:
    if value is None or not _HEX_COLOR.fullmatch(value):
        return
    values[index] = _hex_to_rgb(value)
    save_callback()
    apply_callback()
