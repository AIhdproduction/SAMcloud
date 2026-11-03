"""Dense cloud inspection and semantic class controls."""

import json
import re

from nicegui import ui

from samcloud.app.layout import render_page
from samcloud.app.pointcloud_viewer import default_class_colors, render_pointcloud_viewer
from samcloud.app.state import state
from samcloud.core.paths import CONFIG_DIRECTORY


LAS_CODES = {
    "outdoor": (0, 2, 11, 66, 67, 3, 5, 6, 68, 69, 70, 71, 72, 14, 73, 74, 9),
    "indoor": (0, 2, 75, 6, 76, 77, 78, 79, 80, 81, 82),
}

_COLOR_INPUT_STYLE = """
<style>
.sam-class-color input.q-field__native { color:transparent!important; caret-color:transparent!important; pointer-events:none; }
</style>
"""


def render() -> None:
    """Render RGB/classification mode and per-class visibility controls."""
    def content() -> None:
        ui.add_head_html(_COLOR_INPUT_STYLE)
        project = state.active_project()
        if project is None:
            ui.label("Open a project before viewing dense point cloud results.")
            return
        class_set = project["settings"].get("classification", {}).get("class_set", "outdoor")
        classes = _load_classes(class_set)
        color_hexes = _class_color_hexes(project, len(classes) + 1)
        labels = ["Unknown / noise", *[name.split(",")[0].title() for name in classes]]
        with ui.column().classes("w-full gap-4"):
            with ui.card().classes("w-full"):
                render_pointcloud_viewer(
                    "dense",
                    title="Dense point cloud",
                    class_colors=[_hex_to_rgb(color) for color in color_hexes],
                    class_labels=[
                        f"{LAS_CODES[class_set][class_id]}: {label}"
                        for class_id, label in enumerate(labels)
                    ],
                )
            with ui.card().classes("w-full"):
                ui.label("Classification visibility and colors").classes("text-lg font-medium")
                ui.label(
                    "Switch classes on or off, isolate one class, and edit its color. Color changes are saved in this project."
                ).classes("sam-muted")
                visibility = [True] * len(labels)
                visibility_controls = []

                def apply_visibility() -> None:
                    ui.run_javascript(
                        "window.samcloudPointCloudViewers?.['sam-cloud-dense']?.setClassVisibility("
                        + json.dumps(visibility)
                        + ");"
                    )

                def set_visibility(values: list[bool]) -> None:
                    visibility[:] = values
                    for control, is_visible in zip(visibility_controls, visibility):
                        control.value = is_visible
                    apply_visibility()

                with ui.column().classes("w-full gap-2"):
                    for class_id, label in enumerate(labels):
                        with ui.row().classes("items-center w-full gap-3"):
                            visibility_control = ui.checkbox(
                                f"{LAS_CODES[class_set][class_id]}: {label}",
                                value=True,
                                on_change=lambda event, index=class_id: _set_class_visibility(
                                    visibility, index, event.value, apply_visibility
                                ),
                            ).classes("w-72")
                            visibility_controls.append(visibility_control)
                            ui.button(
                                "Isolate",
                                on_click=lambda index=class_id: set_visibility(
                                    [item == index for item in range(len(labels))]
                                ),
                            ).props("outline dense")
                            ui.color_input(
                                "Color",
                                value=color_hexes[class_id],
                                preview=True,
                                on_change=lambda event, index=class_id: _save_class_color(
                                    index, event.value, class_set, len(labels)
                                ),
                            ).classes("w-40 sam-class-color")
                with ui.row().classes("gap-2"):
                    ui.button("Show all", on_click=lambda: set_visibility([True] * len(labels)))
                    ui.button("Hide all", on_click=lambda: set_visibility([False] * len(labels))).props(
                        "outline"
                    )
                ui.button(
                    "Restore default colors",
                    on_click=lambda: _restore_default_colors(class_set, len(labels)),
                ).props("outline")

    render_page("Dense Cloud", content, current_path="/dense", wide=True)


def _load_classes(class_set: str) -> list[str]:
    data = json.loads((CONFIG_DIRECTORY / "classes.json").read_text(encoding="utf-8"))
    return data[class_set]


def _set_class_visibility(
    visibility: list[bool], class_id: int, value: bool | None, apply_visibility
) -> None:
    """Update one temporary viewer visibility toggle."""
    visibility[class_id] = bool(value)
    apply_visibility()


def _class_color_hexes(project: dict, count: int) -> list[str]:
    """Return saved class colors or the default Cyclone-style palette."""
    defaults = [_rgb_to_hex(color) for color in default_class_colors(count)]
    saved = project.get("settings", {}).get("classification", {}).get("class_colors", {})
    if not isinstance(saved, dict):
        return defaults
    return [
        saved.get(str(index), defaults[index])
        if _is_hex_color(saved.get(str(index), defaults[index]))
        else defaults[index]
        for index in range(count)
    ]


def _save_class_color(class_id: int, color: str | None, class_set: str, count: int) -> None:
    """Persist one class color and update the open dense viewer immediately."""
    if not _is_hex_color(color):
        ui.notify("Select a valid color.", type="negative")
        return
    project_directory = state.active_project_directory
    if project_directory is None:
        return
    project = state.projects.load(project_directory)
    classification = project.setdefault("settings", {}).setdefault("classification", {})
    colors = classification.setdefault("class_colors", {})
    colors[str(class_id)] = color.upper()
    state.projects.save(project_directory, project)
    palette = [_hex_to_rgb(value) for value in _class_color_hexes(project, count)]
    ui.run_javascript(
        "window.samcloudPointCloudViewers?.['sam-cloud-dense']?.setClassColors("
        + json.dumps(palette)
        + ");"
    )


def _restore_default_colors(class_set: str, count: int) -> None:
    """Remove project-specific overrides and reload the standard palette."""
    project_directory = state.active_project_directory
    if project_directory is None:
        return
    project = state.projects.load(project_directory)
    project.setdefault("settings", {}).setdefault("classification", {}).pop("class_colors", None)
    state.projects.save(project_directory, project)
    palette = default_class_colors(count)
    ui.run_javascript(
        "window.samcloudPointCloudViewers?.['sam-cloud-dense']?.setClassColors("
        + json.dumps(palette)
        + ");"
    )
    ui.navigate.to("/dense")


def _is_hex_color(value: object) -> bool:
    return isinstance(value, str) and re.fullmatch(r"#[0-9A-Fa-f]{6}", value) is not None


def _hex_to_rgb(value: str) -> list[int]:
    return [int(value[index : index + 2], 16) for index in (1, 3, 5)]


def _rgb_to_hex(value: list[int]) -> str:
    return "#" + "".join(f"{channel:02X}" for channel in value)
