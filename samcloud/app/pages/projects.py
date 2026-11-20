"""Project creation and reopening page."""

import tempfile
from collections.abc import Iterator, Mapping
from pathlib import Path
from tkinter import TclError, Tk, filedialog
from typing import Any

from nicegui import events, ui

from samcloud.app.i18n import t, translate_or
from samcloud.app.layout import render_page
from samcloud.app.pages.control_points import crs_options, local_crs_hint
from samcloud.app.state import state
from samcloud.classification.custom import standard_classes
from samcloud.core.configuration import load_global_config
from samcloud.core.project_deletion import delete_project
from samcloud.core.storage import DEFAULT_STORAGE


class TranslatedOptions(Mapping):
    """Option mapping with fixed values whose labels resolve in the current UI language on every access.

    Keys (the internal option values) never change; only the displayed labels are translated.
    Pass ``dict(options)`` to ``ui.select`` so the labels are resolved at render time.
    """

    def __init__(self, label_keys: Mapping[Any, str]) -> None:
        self.label_keys = dict(label_keys)

    def __getitem__(self, value: Any) -> str:
        return t(self.label_keys[value])

    def __iter__(self) -> Iterator[Any]:
        return iter(self.label_keys)

    def __len__(self) -> int:
        return len(self.label_keys)

    def __repr__(self) -> str:
        return f"{type(self).__name__}({dict(self)!r})"


STORAGE_OPTIONS = TranslatedOptions({
    "keep_control_point_inputs": "projects.storage.keep_control_point_inputs",
    "keep_classification_inputs": "projects.storage.keep_classification_inputs",
    "keep_refinement_inputs": "projects.storage.keep_refinement_inputs",
    "keep_processing_cache": "projects.storage.keep_processing_cache",
})


ENGINE_OPTIONS = TranslatedOptions({
    "sam3": "projects.engine.sam3",
    "flurseg": "projects.engine.flurseg",
})
CAPTURE_TYPE_OPTIONS = TranslatedOptions({
    "drone_gps": "projects.capture_type.drone_gps",
    "camera": "projects.capture_type.camera",
    "panorama_360": "projects.capture_type.panorama_360",
})
CLASS_SET_OPTIONS = TranslatedOptions({
    "outdoor": "projects.class_set.outdoor",
    "indoor": "projects.class_set.indoor",
})
CONTROL_POINTS_MODE_OPTIONS = TranslatedOptions({
    "auto": "projects.control_points_mode.auto",
    "none": "projects.control_points_mode.none",
    "after_alignment": "projects.control_points_mode.after_alignment",
})


class _ClassEditor:
    """SAM3 class editor on the create-project card: edit standard prompts, add own classes."""

    def __init__(self) -> None:
        self.class_set = ""
        self.engine = ""
        self.prompt_inputs: dict[str, tuple[ui.input, list[str]]] = {}
        self.added: list[dict] = []
        self.flurseg_note = ui.label(
            t("projects.class_editor.flurseg_note", summary=t("projects.class_editor.flurseg_summary"))
        ).classes("text-sm text-slate-400 max-w-[52rem]")
        self.expansion = ui.expansion(t("projects.class_editor.title"), icon="tune").classes("w-full")
        with self.expansion:
            ui.label(t("projects.class_editor.hint")).classes("text-sm text-slate-400")
            self.standard_column = ui.column().classes("w-full gap-1")
            ui.label(t("projects.class_editor.custom_classes")).classes("text-base font-medium q-mt-md")
            self.added_column = ui.column().classes("w-full gap-2")
            ui.button(t("projects.class_editor.add"), icon="add", on_click=self.add_class).props("outline")

    def update(self, class_set: str, engine: str) -> None:
        sam = class_set == "indoor" or engine == "sam3"
        self.flurseg_note.set_visibility(not sam)
        self.expansion.set_visibility(sam)
        if sam and class_set != self.class_set:
            self.class_set = class_set
            self._render_standard()
            self.added.clear()
            self.added_column.clear()
        self.engine = engine

    def _render_standard(self) -> None:
        self.prompt_inputs.clear()
        self.standard_column.clear()
        with self.standard_column:
            for item in standard_classes(self.class_set):
                with ui.row().classes("w-full items-center gap-2 no-wrap"):
                    label = translate_or(f"classes.{item['name']}", item["label"])
                    ui.label(f"{label} ({item['name']})").classes("text-sm w-64 shrink-0")
                    control = ui.input(value=", ".join(item["prompts"])).props("dense").classes("grow")
                self.prompt_inputs[item["name"]] = (control, item["prompts"])

    def add_class(self) -> None:
        entry: dict = {}
        with self.added_column:
            with ui.row().classes("w-full items-center gap-2 no-wrap") as row:
                entry["name"] = ui.input(
                    t("projects.class_editor.name"), placeholder=t("projects.class_editor.name_placeholder")
                ).props("dense").classes("w-44")
                entry["label"] = ui.input(
                    t("projects.class_editor.label"), placeholder=t("projects.class_editor.label_placeholder")
                ).props("dense").classes("w-44")
                entry["prompts"] = ui.input(
                    t("projects.class_editor.prompts"), placeholder=t("projects.class_editor.prompts_placeholder")
                ).props("dense").classes("grow")
                entry["color"] = ui.color_input(t("projects.class_editor.color"), value="#FF00FF").props(
                    "dense").classes("w-36")
                ui.button(icon="delete", on_click=lambda: self._remove(entry, row)).props("flat round dense")
        self.added.append(entry)

    def _remove(self, entry: dict, row) -> None:
        self.added.remove(entry)
        row.delete()

    def customization(self) -> dict | None:
        """Customisation for create_project; None for FlurSeg or when nothing changed."""
        if self.class_set != "indoor" and self.engine != "sam3":
            return None
        prompts = {}
        for name, (control, default) in self.prompt_inputs.items():
            values = [part.strip() for part in (control.value or "").split(",") if part.strip()]
            if values != default:
                prompts[name] = values
        added = [{
            "name": entry["name"].value or "",
            "label": entry["label"].value or "",
            "prompts": entry["prompts"].value or entry["name"].value or "",
            "color": entry["color"].value or "#FF00FF",
        } for entry in self.added]
        return {"prompts": prompts, "added": added} if prompts or added else None


def render() -> None:
    """Render the persistent project manager."""
    def content() -> None:
        panorama_defaults = load_global_config("defaults.yml").get("acquisition", {}).get("panorama", {})
        with ui.card().classes("w-full"):
            ui.label(t("projects.create.title")).classes("text-lg font-medium")
            name = ui.input(t("projects.create.name"), placeholder=t("projects.create.name_placeholder"))
            capture_type = ui.select(
                dict(CAPTURE_TYPE_OPTIONS),
                label=t("projects.create.capture_type"),
                value="drone_gps",
            )
            class_set = ui.select(
                dict(CLASS_SET_OPTIONS),
                label=t("projects.create.class_set"),
                value="outdoor",
            )
            classification_engine = ui.select(
                dict(ENGINE_OPTIONS),
                label=t("projects.create.engine"),
                value="sam3",
            ).classes("min-w-[26rem]").bind_visibility_from(class_set, "value", lambda value: value == "outdoor")
            apriltag_switch = ui.switch(
                t("projects.create.apriltags"), value=class_set.value == "indoor"
            ).mark("apriltag-switch")
            capture_type.on_value_change(
                lambda event: setattr(class_set, "value", "indoor") if event.value == "panorama_360" else None
            )
            class_set.on_value_change(lambda event: setattr(apriltag_switch, "value", event.value == "indoor"))
            tag_upload: dict[str, str | None] = {"path": None}
            with ui.column().classes("w-full gap-2"):
                with ui.column().classes("w-full gap-2").bind_visibility_from(apriltag_switch, "value"):
                    async def receive_tag_coordinates(event: events.UploadEventArguments) -> None:
                        target = Path(tempfile.mkdtemp(prefix="samcloud_apriltags_")) / "apriltags_upload.csv"
                        await event.file.save(target)
                        tag_upload["path"] = str(target)

                    ui.upload(
                        on_upload=receive_tag_coordinates,
                        auto_upload=True,
                        label=t("projects.create.tag_coordinates"),
                    ).props("accept=.csv,.txt").classes("w-full max-w-md").mark("apriltag-coords-upload")
                    tag_crs = ui.select(
                        crs_options(), label=t("projects.create.tag_crs"), value="EPSG:2056+5728"
                    ).classes("w-full max-w-md").mark("apriltag-crs")
                    tag_crs_hint = ui.label(local_crs_hint()).classes("sam-muted text-sm").mark("apriltag-crs-local-hint")
                    tag_crs_hint.set_visibility(False)
                    tag_crs.on_value_change(lambda event: tag_crs_hint.set_visibility(event.value == "LOCAL"))
                    ui.label(t("projects.create.tag_hint")).classes("sam-muted text-sm")
            class_editor = _ClassEditor()

            def update_class_editor() -> None:
                class_editor.update(class_set.value or "outdoor", classification_engine.value or "sam3")

            class_set.on_value_change(lambda _: update_class_editor())
            classification_engine.on_value_change(lambda _: update_class_editor())
            update_class_editor()
            control_points_mode = ui.select(
                dict(CONTROL_POINTS_MODE_OPTIONS),
                label=t("projects.create.control_points"),
                value="auto",
            )
            auto_dxf = ui.switch(t("projects.create.auto_dxf"), value=True).mark("auto-dxf-switch")
            with ui.card().classes("w-full"):
                ui.label(t("projects.create.storage")).classes("text-base font-medium")
                storage_switches = {
                    key: ui.switch(label, value=DEFAULT_STORAGE[key])
                    for key, label in STORAGE_OPTIONS.items()
                }

            with ui.column().classes("w-full gap-3").bind_visibility_from(
                capture_type, "value", lambda value: value in {"drone_gps", "camera"}
            ):
                image_directory = ui.input(
                    t("projects.create.image_directory"), placeholder="D:\\data\\camera-images"
                ).classes("w-full")
                rtk_enabled = ui.switch(t("projects.create.rtk"), value=True)

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
                            title=t("projects.create.select_directory"),
                            mustexist=True,
                        )
                    except TclError as error:
                        ui.notify(t("projects.create.dialog_failed", error=error), type="negative")
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
                    ).props(f"flat round dense aria-label='{t('projects.create.select_directory')}'")

                def update_rtk_option() -> None:
                    is_rtk_capture = capture_type.value == "drone_gps"
                    rtk_enabled.set_visibility(is_rtk_capture)
                    rtk_enabled.value = is_rtk_capture

                capture_type.on_value_change(lambda _: update_rtk_option())
                update_rtk_option()

            with ui.column().classes("w-full gap-3").bind_visibility_from(
                capture_type, "value", lambda value: value == "panorama_360"
            ):
                ui.label(t("projects.create.panorama_hint")).classes("text-sm")
                panorama_video_path = ui.input(
                    t("projects.create.panorama_video"), placeholder="D:\\data\\tour.mp4"
                ).classes("w-full")
                panorama_image_directory = ui.input(
                    t("projects.create.panorama_directory"),
                    placeholder="D:\\data\\insta360-photos",
                ).classes("w-full")
                frame_interval_seconds = ui.number(
                    t("projects.create.frame_interval"),
                    value=panorama_defaults.get("frame_interval_seconds", 1.0),
                    min=0.1,
                    step=0.1,
                ).classes("w-full")

            def create_project() -> None:
                try:
                    if capture_type.value is None:
                        raise ValueError(t("projects.create.select_capture_type"))
                    if class_set.value is None:
                        raise ValueError(t("projects.create.select_class_set"))
                    project_directory = state.projects.create_project(
                        name.value,
                        image_directory.value or "",
                        capture_type=capture_type.value,
                        class_set=class_set.value,
                        classification_engine=(classification_engine.value or "sam3")
                        if class_set.value == "outdoor" else "sam3",
                        custom_classes=class_editor.customization(),
                        panorama_image_directory=panorama_image_directory.value or "",
                        panorama_video_path=panorama_video_path.value or "",
                        frame_interval_seconds=float(frame_interval_seconds.value or 0),
                        rtk_enabled=rtk_enabled.value,
                        control_points_mode=control_points_mode.value,
                        storage={key: bool(control.value) for key, control in storage_switches.items()},
                        auto_dxf=bool(auto_dxf.value),
                        apriltags_enabled=bool(apriltag_switch.value),
                        apriltag_coordinates_csv=tag_upload["path"] if apriltag_switch.value else None,
                        apriltag_crs=tag_crs.value if tag_upload["path"] else None,
                    )
                    state.open_project(project_directory)
                    ui.notify(t("projects.create.created", name=project_directory.name))
                    ui.navigate.to("/")
                except (FileExistsError, ValueError) as error:
                    ui.notify(str(error), type="negative")

            ui.button(t("projects.create.button"), on_click=create_project)

        ui.label(t("projects.existing.title")).classes("text-lg font-medium")

        @ui.refreshable
        def render_existing_projects() -> None:
            projects = state.projects.discover_projects()
            if not projects:
                ui.label(t("projects.existing.empty"))
                return
            for project_directory in projects:
                project = state.projects.load(project_directory)
                with ui.card().classes("w-full"):
                    with ui.row().classes("items-center justify-between w-full"):
                        with ui.column().classes("gap-0"):
                            ui.label(project["project"]["name"]).classes("font-medium")
                            ui.label(str(project_directory)).classes("text-xs text-slate-500")
                            acquisition = project.get("acquisition", {})
                            capture_type_value = acquisition.get("type")
                            capture_label = (
                                CAPTURE_TYPE_OPTIONS[capture_type_value]
                                if capture_type_value in CAPTURE_TYPE_OPTIONS
                                else t("projects.capture_type.legacy")
                            )
                            ui.label(capture_label).classes("text-xs text-slate-500")
                        with ui.row().classes("items-center gap-2"):
                            ui.button(
                                t("common.open"),
                                on_click=lambda path=project_directory: _open_project(path),
                            )
                            ui.button(
                                t("common.delete"),
                                icon="delete",
                                on_click=lambda path=project_directory: _confirm_delete(
                                    path, render_existing_projects.refresh
                                ),
                            ).props("outline color=negative")

        render_existing_projects()

    render_page(t("nav.projects"), content, current_path="/projects")


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
        ui.notify(t("projects.delete.busy"), type="warning")
        return

    with ui.dialog() as dialog, ui.card().classes("w-full"):
        ui.label(t("projects.delete.title")).classes("text-lg font-medium")
        ui.label(t("projects.delete.message", name=project_directory.name)).classes("text-sm")
        with ui.row().classes("justify-end w-full gap-2 q-mt-md"):
            ui.button(t("common.cancel"), on_click=dialog.close).props("flat")

            def delete_confirmed() -> None:
                try:
                    delete_project(state.projects.projects_directory, project_directory)
                    if state.active_project_directory == project_directory.resolve():
                        state.active_project_directory = None
                    dialog.close()
                    refresh_projects()
                    ui.notify(t("projects.delete.done", name=project_directory.name))
                except (OSError, ValueError) as error:
                    ui.notify(str(error), type="negative")

            ui.button(t("projects.delete.confirm"), icon="delete", on_click=delete_confirmed).props(
                "color=negative"
            )
    dialog.open()
