"""Project creation and reopening page."""

from pathlib import Path
from tkinter import TclError, Tk, filedialog

from nicegui import ui

from samcloud.app.layout import render_page
from samcloud.app.state import state
from samcloud.classification.custom import standard_classes
from samcloud.core.configuration import load_global_config
from samcloud.core.project_deletion import delete_project
from samcloud.core.storage import DEFAULT_STORAGE

STORAGE_OPTIONS = {
    "keep_control_point_inputs": "Passpunkte später neu optimieren können",
    "keep_classification_inputs": "Klassifizierung später neu rechnen können",
    "keep_refinement_inputs": "Geometrie-Verfeinerung später neu rechnen können",
    "keep_processing_cache": "Schritt 6 (Verbessern & filtern) später schnell neu rechnen können – "
                             "behält Tiefenkarten, SAM-Bilder und Depth-Anything (braucht viel Speicher)",
}


ENGINE_OPTIONS = {
    "sam3": "SAM3 – Klassen anpassbar und erweiterbar (Standard)",
    "flurseg": "FlurSeg – ca. 15× schneller, 11 feste Klassen",
}
FLURSEG_CLASS_SUMMARY = (
    "Boden, versiegelte Fläche (Strasse, Gehweg, Markierung → road), Randstein, niedrige Vegetation, "
    "Baum, Gebäude, Fahrzeug, Baustelle (→ construction machinery), Schachtdeckel (→ manhole); "
    "alles andere → unknown"
)


class _ClassEditor:
    """SAM3 class editor on the create-project card: edit standard prompts, add own classes."""

    def __init__(self) -> None:
        self.class_set = ""
        self.engine = ""
        self.prompt_inputs: dict[str, tuple[ui.input, list[str]]] = {}
        self.added: list[dict] = []
        self.flurseg_note = ui.label(
            "FlurSeg ist etwa 15× schneller als SAM3, die Klassen sind aber durch das Training fest "
            f"und nicht anpassbar: {FLURSEG_CLASS_SUMMARY}."
        ).classes("text-sm text-slate-400 max-w-[52rem]")
        self.expansion = ui.expansion("Klassen anpassen (SAM3)", icon="tune").classes("w-full")
        with self.expansion:
            ui.label(
                "Prompts sind englische Begriffe, mit Komma getrennt. Die Standardklassen bleiben erhalten "
                "(Schritt 6 und die Geometrie-Regeln nutzen sie), eigene Klassen werden hinten angehängt "
                "und bekommen LAS-Codes ab 64."
            ).classes("text-sm text-slate-400")
            self.standard_column = ui.column().classes("w-full gap-1")
            ui.label("Eigene Klassen").classes("text-base font-medium q-mt-md")
            self.added_column = ui.column().classes("w-full gap-2")
            ui.button("Klasse hinzufügen", icon="add", on_click=self.add_class).props("outline")

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
                    ui.label(f"{item['label']} ({item['name']})").classes("text-sm w-64 shrink-0")
                    control = ui.input(value=", ".join(item["prompts"])).props("dense").classes("grow")
                self.prompt_inputs[item["name"]] = (control, item["prompts"])

    def add_class(self) -> None:
        entry: dict = {}
        with self.added_column:
            with ui.row().classes("w-full items-center gap-2 no-wrap") as row:
                entry["name"] = ui.input("Name (englisch)", placeholder="solar panel").props("dense").classes("w-44")
                entry["label"] = ui.input("Bezeichnung", placeholder="Solaranlage").props("dense").classes("w-44")
                entry["prompts"] = ui.input("Prompts", placeholder="solar panel, photovoltaic module").props(
                    "dense").classes("grow")
                entry["color"] = ui.color_input("Farbe", value="#FF00FF").props("dense").classes("w-36")
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
            classification_engine = ui.select(
                ENGINE_OPTIONS,
                label="Klassifizierungsmodell (Aussenbereich)",
                value="sam3",
            ).classes("min-w-[26rem]").bind_visibility_from(class_set, "value", lambda value: value == "outdoor")
            class_editor = _ClassEditor()

            def update_class_editor() -> None:
                class_editor.update(class_set.value or "outdoor", classification_engine.value or "sam3")

            class_set.on_value_change(lambda _: update_class_editor())
            classification_engine.on_value_change(lambda _: update_class_editor())
            update_class_editor()
            control_points_mode = ui.select(
                {
                    "none": "No control points - continue automatically",
                    "after_alignment": "Use control points after alignment",
                },
                label="Ground control points",
                value="none",
            )
            with ui.card().classes("w-full"):
                ui.label("Speicher").classes("text-base font-medium")
                storage_switches = {
                    key: ui.switch(label, value=DEFAULT_STORAGE[key])
                    for key, label in STORAGE_OPTIONS.items()
                }

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
                        classification_engine=(classification_engine.value or "sam3")
                        if class_set.value == "outdoor" else "sam3",
                        custom_classes=class_editor.customization(),
                        panorama_image_directory=panorama_image_directory.value or "",
                        panorama_video_path=panorama_video_path.value or "",
                        frame_interval_seconds=float(frame_interval_seconds.value or 0),
                        rtk_enabled=rtk_enabled.value,
                        control_points_mode=control_points_mode.value,
                        storage={key: bool(control.value) for key, control in storage_switches.items()},
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
