"""Project coordinate and processing settings page."""

from typing import Any

from nicegui import app, ui

from samcloud.acquisition import (
    CaptureValidationError,
    prepare_project_panorama,
    validate_capture_configuration,
)
from samcloud.app.i18n import (
    DEFAULT_LANGUAGE,
    SUPPORTED_LANGUAGES,
    clear_user_language,
    config_language,
    current_language,
    set_language,
    t,
    user_language,
)
from samcloud.app.layout import render_page
from samcloud.app.pages.projects import CAPTURE_TYPE_OPTIONS, CLASS_SET_OPTIONS, ENGINE_OPTIONS, TranslatedOptions
from samcloud.app.state import state
from samcloud.share.ui import render_settings_card as render_share_settings_card
from samcloud.core.configuration import load_export_crs_presets
from samcloud.core.storage import storage_settings
from samcloud.app.pages.projects import STORAGE_OPTIONS
from samcloud.core.storage import DEFAULT_STORAGE


CONTROL_POINTS_MODE_OPTIONS = TranslatedOptions({
    "auto": "settings.control_points_mode.auto",
    "none": "settings.control_points_mode.none",
    "after_alignment": "settings.control_points_mode.after_alignment",
    "automatic": "settings.control_points_mode.automatic",
})
CLASSIFICATION_MODE_OPTIONS = TranslatedOptions({
    "legacy": "settings.classification.mode_legacy",
    "segearth_ov3": "settings.classification.mode_segearth_ov3",
})
TILE_SIZE_OPTIONS = TranslatedOptions({
    768: "settings.classification.tile_768",
    1008: "settings.classification.tile_1008",
    1280: "settings.classification.tile_1280",
})
DEPTH_MODEL_OPTIONS = TranslatedOptions({
    "large": "settings.refinement.depth_large",
    "small": "settings.refinement.depth_small",
    "off": "settings.refinement.depth_off",
})

# Unsaved form values survive the page reload that applies a new UI language.
DRAFT_STORAGE_KEY = "settings_draft"


def _export_crs_options() -> dict[str, str]:
    """Return settings select options from the central export CRS configuration."""
    return {key: str(preset["label"]) for key, preset in load_export_crs_presets().items()}


def _snapshot_draft(controls: dict[str, Any]) -> dict[str, Any]:
    """Return the current (possibly unsaved) value of every registered settings control."""
    return {key: control.value for key, control in controls.items() if control is not None}


def _apply_draft(controls: dict[str, Any], draft: Any, owner: str) -> None:
    """Put snapshot values back into the freshly rendered controls; unknown keys and invalid values are skipped.

    A draft only applies to the project it was taken from (``owner``).
    """
    if not isinstance(draft, dict) or draft.get("project") != owner or not isinstance(draft.get("values"), dict):
        return
    for key, value in draft["values"].items():
        control = controls.get(key)
        if control is None:
            continue
        try:
            control.value = value
        except Exception:  # e.g. an option that no longer exists; keep the rendered value
            continue


def _reload_keeping_draft(controls: dict[str, Any], owner: str) -> None:
    """Store the unsaved form values in the tab storage, then reload the page in the new language."""
    try:
        app.storage.tab[DRAFT_STORAGE_KEY] = {"project": owner, "values": _snapshot_draft(controls)}
    except Exception:  # no client connection or tab storage not ready: reload without a draft
        pass
    ui.navigate.reload()


def _render_language_card(controls: dict[str, Any], owner: str) -> None:
    """Language selector: the choice is stored immediately and the page re-renders in the new language."""
    with ui.card().classes("w-full"):
        ui.label(t("settings.language.title")).classes("text-lg font-medium")

        def change_language(event: Any) -> None:
            value = event.value
            if not value or value == current_language():
                return
            try:
                set_language(value)
            except (OSError, ValueError) as error:
                ui.notify(t("settings.language.failed", error=error), type="negative")
                return
            _reload_keeping_draft(controls, owner)

        def reset_language() -> None:
            try:
                clear_user_language()
            except OSError as error:
                ui.notify(t("settings.language.failed", error=error), type="negative")
                return
            _reload_keeping_draft(controls, owner)

        ui.select(
            dict(SUPPORTED_LANGUAGES),
            label=t("settings.language.label"),
            value=current_language(),
            on_change=change_language,
        ).classes("min-w-[16rem]").mark("settings-language")
        default_language = config_language() or DEFAULT_LANGUAGE
        ui.label(
            t("settings.language.config_default", language=SUPPORTED_LANGUAGES.get(default_language, default_language))
        ).classes("sam-muted text-sm")
        ui.label(t("settings.language.hint")).classes("sam-muted text-sm")
        reset = ui.button(t("settings.language.reset"), on_click=reset_language).props("flat no-caps")
        reset.mark("settings-language-reset")
        reset.set_enabled(user_language() is not None)


def render() -> None:
    """Render editable coordinate system and workflow settings."""
    def content() -> None:
        draft_controls: dict[str, Any] = {}
        project_directory = state.active_project_directory
        draft_owner = str(project_directory)
        _render_language_card(draft_controls, draft_owner)
        render_share_settings_card()
        project = state.active_project()
        if project_directory is None or project is None:
            ui.label(t("settings.no_project"))
            return

        settings = project["settings"]
        workflow = settings["workflow"]
        coordinates = settings["coordinate_systems"]
        classification = settings.setdefault("classification", {"class_set": "outdoor"})
        classification.setdefault("geometry_refinement", True)
        classification.setdefault("min_votes", 1)
        classification.setdefault("tile_size", 1008)
        classification.setdefault("tile_min_overlap", 0.2)
        classification.setdefault("mode", "segearth_ov3")
        classification.setdefault("engine", "sam3")
        classification.setdefault("background_threshold", 0.4)
        classification.setdefault("query_threshold", 0.4)
        classification.setdefault("threshold_sweep", [])
        refinement = settings.setdefault("refinement", {})
        refinement.setdefault("depth_model", "large")
        refinement.setdefault("depth_fill", True)
        refinement.setdefault("surface_fusion", True)
        storage = settings.setdefault("storage", storage_settings(settings))
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
        capture_label = (
            CAPTURE_TYPE_OPTIONS[capture_type] if capture_type in CAPTURE_TYPE_OPTIONS else t("common.unknown")
        )
        control_points_mode = project.get("workflow", {}).get(
            "control_points_mode",
            "auto",
        )

        with ui.card().classes("w-full"):
            ui.label(t("settings.capture.title")).classes("text-lg font-medium")
            ui.label(t("settings.capture.type", type=capture_label))
            if acquisition_locked:
                ui.label(t("settings.capture.locked")).classes("text-sm")
            if capture_type in {"drone_gps", "camera"}:
                image_directory = ui.input(
                    t("settings.capture.image_directory"), value=sources.get("image_directory", "")
                ).classes("w-full")
                image_directory.set_enabled(not acquisition_locked)
                rtk_enabled = ui.switch(
                    t("settings.capture.rtk"), value=bool(georeferencing.get("rtk_enabled", False))
                )
                rtk_enabled.set_visibility(capture_type == "drone_gps")
                rtk_enabled.set_enabled(not acquisition_locked)
                panorama_video_path = None
                panorama_image_directory = None
                frame_interval_seconds = None
            else:
                panorama_video_path = ui.input(
                    t("settings.capture.panorama_video"), value=sources.get("panorama_video_path", "")
                ).classes("w-full")
                panorama_image_directory = ui.input(
                    t("settings.capture.panorama_directory"),
                    value=sources.get("panorama_image_directory", ""),
                ).classes("w-full")
                frame_interval_seconds = ui.number(
                    t("settings.capture.frame_interval"),
                    value=panorama.get("frame_interval_seconds", 1.0),
                    min=0.1,
                    step=0.1,
                ).classes("w-full")
                for element in (panorama_video_path, panorama_image_directory, frame_interval_seconds):
                    element.set_enabled(not acquisition_locked)
                image_directory = None
                rtk_enabled = None
        with ui.card().classes("w-full"):
            ui.label(t("settings.coordinates.title")).classes("text-lg font-medium")
            input_crs = ui.input(t("settings.coordinates.input"), value=coordinates["input_crs"])
            working_crs = ui.input(t("settings.coordinates.working"), value=coordinates["working_crs"])
            export_crs_options = _export_crs_options()
            export_crs_value = str(coordinates.get("export_crs", "EPSG:2056")).upper()
            if export_crs_value not in export_crs_options:
                export_crs_value = "EPSG:2056"
            export_crs = ui.select(
                export_crs_options,
                label=t("settings.coordinates.export"),
                value=export_crs_value,
            )
            ui.label(t("settings.coordinates.hint")).classes("text-sm")
        with ui.card().classes("w-full"):
            ui.label(t("settings.processing.title")).classes("text-lg font-medium")
            use_gpu = ui.switch(t("settings.processing.use_gpu"), value=workflow["use_gpu"])
            auto_dxf = ui.switch(t("settings.processing.auto_dxf"), value=bool(workflow.get("auto_dxf", True))).mark("auto-dxf-switch")
            control_points = ui.select(
                dict(CONTROL_POINTS_MODE_OPTIONS),
                label=t("settings.processing.control_points"),
                value=control_points_mode,
            )
            control_points.set_enabled(not acquisition_locked)
        with ui.card().classes("w-full"):
            ui.label(t("settings.storage.title")).classes("text-lg font-medium")
            storage_switches = {
                key: ui.switch(label, value=bool(storage.get(key, DEFAULT_STORAGE.get(key, True))))
                for key, label in STORAGE_OPTIONS.items()
            }
        with ui.card().classes("w-full"):
            ui.label(t("settings.classification.title")).classes("text-lg font-medium")
            ui.label(t("settings.classification.hint")).classes("text-sm")
            class_set = ui.select(
                dict(CLASS_SET_OPTIONS),
                value=classification["class_set"],
                label=t("settings.classification.class_set"),
            )
            min_votes = ui.number(
                t("settings.classification.min_votes"),
                value=int(classification["min_votes"]),
                min=1,
                max=20,
                step=1,
            )
            engine = ui.select(
                dict(ENGINE_OPTIONS),
                value=classification["engine"] if classification["engine"] in {"sam3", "flurseg"} else "sam3",
                label=t("settings.classification.engine"),
            ).classes("w-full")
            engine_hint = ui.label(t("settings.classification.engine_hint")).classes("text-sm")

            def update_engine_visibility() -> None:
                outdoor = class_set.value == "outdoor"
                engine.set_visibility(outdoor)
                engine_hint.set_visibility(not outdoor)

            class_set.on_value_change(lambda _event: update_engine_visibility())
            update_engine_visibility()
            classification_mode = ui.select(
                dict(CLASSIFICATION_MODE_OPTIONS),
                value=classification["mode"],
                label=t("settings.classification.mode"),
            )
            background_threshold = ui.number(
                t("settings.classification.background_threshold"),
                value=float(classification["background_threshold"]),
                min=0.0,
                max=1.0,
                step=0.05,
                format="%.2f",
            )
            tile_size = ui.select(
                dict(TILE_SIZE_OPTIONS),
                value=int(classification["tile_size"]),
                label=t("settings.classification.tile_size"),
            )
            tile_min_overlap = ui.number(
                t("settings.classification.tile_overlap"),
                value=float(classification["tile_min_overlap"]),
                min=0.0,
                max=0.8,
                step=0.05,
                format="%.2f",
            )
            geometry_refinement = ui.switch(
                t("settings.classification.geometry_refinement"),
                value=bool(classification["geometry_refinement"]),
            )
            ui.label(t("settings.refinement.title")).classes("text-lg font-medium q-mt-md")
            ui.label(t("settings.refinement.hint")).classes("text-sm")
            depth_model = ui.select(
                dict(DEPTH_MODEL_OPTIONS),
                label=t("settings.refinement.depth_model"),
                value=str(refinement.get("depth_model", "large")),
            ).classes("w-full")
            depth_fill = ui.switch(
                t("settings.refinement.depth_fill"),
                value=bool(refinement.get("depth_fill", True)),
            )
            surface_fusion = ui.switch(
                t("settings.refinement.surface_fusion"),
                value=bool(refinement.get("surface_fusion", True)),
            )

        draft_controls.update({
            "image_directory": image_directory,
            "rtk_enabled": rtk_enabled,
            "panorama_video_path": panorama_video_path,
            "panorama_image_directory": panorama_image_directory,
            "frame_interval_seconds": frame_interval_seconds,
            "input_crs": input_crs,
            "working_crs": working_crs,
            "export_crs": export_crs,
            "use_gpu": use_gpu,
            "auto_dxf": auto_dxf,
            "control_points": control_points,
            "class_set": class_set,
            "min_votes": min_votes,
            "engine": engine,
            "classification_mode": classification_mode,
            "background_threshold": background_threshold,
            "tile_size": tile_size,
            "tile_min_overlap": tile_min_overlap,
            "geometry_refinement": geometry_refinement,
            "depth_model": depth_model,
            "depth_fill": depth_fill,
            "surface_fusion": surface_fusion,
            **{f"storage.{key}": control for key, control in storage_switches.items()},
        })  # None entries (controls of the other capture type) are skipped by snapshot and restore

        async def restore_draft() -> None:
            """Re-apply unsaved values kept across a language reload (tab storage needs a connected client)."""
            try:
                await ui.context.client.connected(timeout=10)
                draft = app.storage.tab.pop(DRAFT_STORAGE_KEY, None)
            except Exception:  # no connection or tab storage unavailable: keep the saved values
                return
            _apply_draft(draft_controls, draft, draft_owner)

        ui.timer(0, restore_draft, once=True)

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
                workflow["use_gpu"] = use_gpu.value
                workflow["auto_dxf"] = bool(auto_dxf.value)
                workflow["control_points_enabled"] = control_points.value in {"after_alignment", "automatic", "auto"}
                project.setdefault("workflow", {})["control_points_mode"] = control_points.value
                classification["class_set"] = class_set.value
                classification["min_votes"] = int(min_votes.value or 0)
                classification["mode"] = classification_mode.value
                classification["engine"] = engine.value if class_set.value == "outdoor" else "sam3"
                if classification["engine"] not in {"sam3", "flurseg"}:
                    raise ValueError(t("settings.errors.unsupported_engine"))
                classification["background_threshold"] = float(background_threshold.value or 0)
                classification["tile_size"] = int(tile_size.value or 0)
                classification["tile_min_overlap"] = float(tile_min_overlap.value or 0)
                classification["geometry_refinement"] = bool(geometry_refinement.value)
                if depth_model.value not in {"large", "small", "off"}:
                    raise ValueError(t("settings.errors.unsupported_depth_model"))
                refinement["depth_model"] = depth_model.value
                refinement["depth_fill"] = bool(depth_fill.value)
                refinement["surface_fusion"] = bool(surface_fusion.value)
                storage.update({key: bool(control.value) for key, control in storage_switches.items()})
                if classification["min_votes"] < 1:
                    raise ValueError(t("settings.errors.min_votes"))
                if classification["mode"] not in {"legacy", "segearth_ov3"}:
                    raise ValueError(t("settings.errors.unsupported_mode"))
                if not 0.0 <= classification["background_threshold"] <= 1.0:
                    raise ValueError(t("settings.errors.background_threshold"))
                if classification["tile_size"] not in {768, 1008, 1280}:
                    raise ValueError(t("settings.errors.tile_size"))
                if not 0.0 <= classification["tile_min_overlap"] <= 0.8:
                    raise ValueError(t("settings.errors.tile_overlap"))
                state.projects.save(project_directory, project)
                ui.notify(t("settings.saved"))
            except (CaptureValidationError, ValueError) as error:
                ui.notify(str(error), type="negative")

        ui.button(t("settings.save"), on_click=save_settings)

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
                    ui.notify(t(
                        "settings.panorama.prepared",
                        panoramas=prepared.panorama_count,
                        images=prepared.cubemap_image_count,
                    ))
                    ui.navigate.to("/")
                except (CaptureValidationError, ValueError) as error:
                    ui.notify(str(error), type="negative")

            ui.button(t("settings.panorama.prepare"), on_click=prepare_inputs).props("outline")

    render_page(t("nav.settings"), content, current_path="/settings")
