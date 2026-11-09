"""Project coordinate and processing settings page."""

from nicegui import ui

from samcloud.acquisition import (
    CaptureValidationError,
    prepare_project_panorama,
    validate_capture_configuration,
)
from samcloud.app.layout import render_page
from samcloud.app.pages.projects import ENGINE_OPTIONS
from samcloud.app.state import state
from samcloud.core.configuration import load_export_crs_presets
from samcloud.core.storage import storage_settings
from samcloud.app.pages.projects import STORAGE_OPTIONS
from samcloud.core.storage import DEFAULT_STORAGE


CONTROL_POINTS_MODE_OPTIONS = {
    "none": "No control points - continue automatically",
    "after_alignment": "Use control points after alignment",
    "automatic": "Automatic (detect targets)",
}


def _export_crs_options() -> dict[str, str]:
    """Return settings select options from the central export CRS configuration."""
    return {key: str(preset["label"]) for key, preset in load_export_crs_presets().items()}


def render() -> None:
    """Render editable coordinate system and workflow settings."""
    def content() -> None:
        project_directory = state.active_project_directory
        project = state.active_project()
        if project_directory is None or project is None:
            ui.label("Open a project before editing settings.")
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
        capture_label = {
            "drone_gps": "Drone with GPS / RTK",
            "camera": "Standard camera without GPS",
            "panorama_360": "360 degree camera / Insta360",
        }.get(capture_type, "Unknown")
        control_points_mode = project.get("workflow", {}).get(
            "control_points_mode",
            "after_alignment" if workflow.get("control_points_enabled", False) else "none",
        )

        with ui.card().classes("w-full"):
            ui.label("Capture sources").classes("text-lg font-medium")
            ui.label(f"Capture type: {capture_label}")
            if acquisition_locked:
                ui.label("Capture settings are locked because processing has started.").classes("text-sm")
            if capture_type in {"drone_gps", "camera"}:
                image_directory = ui.input(
                    "Image directory", value=sources.get("image_directory", "")
                ).classes("w-full")
                image_directory.set_enabled(not acquisition_locked)
                rtk_enabled = ui.switch(
                    "Images use RTK GPS", value=bool(georeferencing.get("rtk_enabled", False))
                )
                rtk_enabled.set_visibility(capture_type == "drone_gps")
                rtk_enabled.set_enabled(not acquisition_locked)
                panorama_video_path = None
                panorama_image_directory = None
                frame_interval_seconds = None
            else:
                panorama_video_path = ui.input(
                    "360 degree MP4 video", value=sources.get("panorama_video_path", "")
                ).classes("w-full")
                panorama_image_directory = ui.input(
                    "360 degree JPG panorama directory",
                    value=sources.get("panorama_image_directory", ""),
                ).classes("w-full")
                frame_interval_seconds = ui.number(
                    "Video frame interval in seconds",
                    value=panorama.get("frame_interval_seconds", 1.0),
                    min=0.1,
                    step=0.1,
                ).classes("w-full")
                for element in (panorama_video_path, panorama_image_directory, frame_interval_seconds):
                    element.set_enabled(not acquisition_locked)
                image_directory = None
                rtk_enabled = None
        with ui.card().classes("w-full"):
            ui.label("Coordinate systems").classes("text-lg font-medium")
            input_crs = ui.input("Image/GNSS CRS", value=coordinates["input_crs"])
            working_crs = ui.input("Working CRS", value=coordinates["working_crs"])
            export_crs_options = _export_crs_options()
            export_crs_value = str(coordinates.get("export_crs", "EPSG:2056")).upper()
            if export_crs_value not in export_crs_options:
                export_crs_value = "EPSG:2056"
            export_crs = ui.select(
                export_crs_options,
                label="Export CRS",
                value=export_crs_value,
            )
            ui.label("Choose the output coordinate system configured in config/export_crs.yml.").classes("text-sm")
        with ui.card().classes("w-full"):
            ui.label("Processing settings").classes("text-lg font-medium")
            use_gpu = ui.switch("Use GPU", value=workflow["use_gpu"])
            control_points = ui.select(
                CONTROL_POINTS_MODE_OPTIONS,
                label="Ground control points",
                value=control_points_mode,
            )
            control_points.set_enabled(not acquisition_locked)
        with ui.card().classes("w-full"):
            ui.label("Speicher").classes("text-lg font-medium")
            storage_switches = {
                key: ui.switch(label, value=bool(storage.get(key, DEFAULT_STORAGE.get(key, True))))
                for key, label in STORAGE_OPTIONS.items()
            }
        with ui.card().classes("w-full"):
            ui.label("Classification").classes("text-lg font-medium")
            ui.label("These values apply to the next classification run.").classes("text-sm")
            class_set = ui.select(
                {"outdoor": "Outdoor", "indoor": "Indoor"},
                value=classification["class_set"],
                label="Semantic class set",
            )
            min_votes = ui.number(
                "Minimum agreeing images per point",
                value=int(classification["min_votes"]),
                min=1,
                max=20,
                step=1,
            )
            engine = ui.select(
                ENGINE_OPTIONS,
                value=classification["engine"] if classification["engine"] in {"sam3", "flurseg"} else "sam3",
                label="Klassifizierungsmodell (Aussenbereich)",
            ).classes("w-full")
            engine_hint = ui.label("Innenbereich wird immer mit SAM3 klassifiziert.").classes("text-sm")

            def update_engine_visibility() -> None:
                outdoor = class_set.value == "outdoor"
                engine.set_visibility(outdoor)
                engine_hint.set_visibility(not outdoor)

            class_set.on_value_change(lambda _event: update_engine_visibility())
            update_engine_visibility()
            classification_mode = ui.select(
                {
                    "legacy": "Legacy (instance masks, majority vote)",
                    "segearth_ov3": "SegEarth-OV3 (head fusion, presence gate, weighted voting)",
                },
                value=classification["mode"],
                label="Classification mode",
            )
            background_threshold = ui.number(
                "SegEarth-OV3 background threshold",
                value=float(classification["background_threshold"]),
                min=0.0,
                max=1.0,
                step=0.05,
                format="%.2f",
            )
            tile_size = ui.select(
                {
                    768: "768 px - faster, lower detail",
                    1008: "1008 px - balanced",
                    1280: "1280 px - higher detail, more GPU memory",
                },
                value=int(classification["tile_size"]),
                label="SAM3 tile size",
            )
            tile_min_overlap = ui.number(
                "Tile overlap",
                value=float(classification["tile_min_overlap"]),
                min=0.0,
                max=0.8,
                step=0.05,
                format="%.2f",
            )
            geometry_refinement = ui.switch(
                "Geometric refinement (height & flatness check)",
                value=bool(classification["geometry_refinement"]),
            )
            ui.label("Improve & filter (step 6)").classes("text-lg font-medium q-mt-md")
            ui.label(
                "Indoor only: Depth-Anything-V2 predicts depth where COLMAP finds no texture (white walls) "
                "and helps to remove floating points and to fill floors, ceilings and walls."
            ).classes("text-sm")
            depth_model = ui.select(
                {
                    "large": "Depth-Anything-V2 Large (best, CC-BY-NC-4.0 weights)",
                    "small": "Depth-Anything-V2 Small (faster, Apache-2.0 weights)",
                    "off": "Off",
                },
                label="Depth model",
                value=str(refinement.get("depth_model", "large")),
            ).classes("w-full")
            depth_fill = ui.switch(
                "Lücken mit Depth-Anything füllen (nur wo es mit den COLMAP-Tiefenkarten übereinstimmt)",
                value=bool(refinement.get("depth_fill", True)),
            )
            surface_fusion = ui.switch(
                "Oberflächen zusammenführen: alle Tiefenkarten zu einer Punktlage (Wand, Decke, Boden je eine Ebene)",
                value=bool(refinement.get("surface_fusion", True)),
            )

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
                workflow["control_points_enabled"] = control_points.value in {"after_alignment", "automatic"}
                project.setdefault("workflow", {})["control_points_mode"] = control_points.value
                classification["class_set"] = class_set.value
                classification["min_votes"] = int(min_votes.value or 0)
                classification["mode"] = classification_mode.value
                classification["engine"] = engine.value if class_set.value == "outdoor" else "sam3"
                if classification["engine"] not in {"sam3", "flurseg"}:
                    raise ValueError("Select a supported classification model")
                classification["background_threshold"] = float(background_threshold.value or 0)
                classification["tile_size"] = int(tile_size.value or 0)
                classification["tile_min_overlap"] = float(tile_min_overlap.value or 0)
                classification["geometry_refinement"] = bool(geometry_refinement.value)
                if depth_model.value not in {"large", "small", "off"}:
                    raise ValueError("Select a supported depth model")
                refinement["depth_model"] = depth_model.value
                refinement["depth_fill"] = bool(depth_fill.value)
                refinement["surface_fusion"] = bool(surface_fusion.value)
                storage.update({key: bool(control.value) for key, control in storage_switches.items()})
                if classification["min_votes"] < 1:
                    raise ValueError("Minimum agreeing images must be at least 1")
                if classification["mode"] not in {"legacy", "segearth_ov3"}:
                    raise ValueError("Select a supported classification mode")
                if not 0.0 <= classification["background_threshold"] <= 1.0:
                    raise ValueError("Background threshold must be between 0.00 and 1.00")
                if classification["tile_size"] not in {768, 1008, 1280}:
                    raise ValueError("Select a supported SAM3 tile size")
                if not 0.0 <= classification["tile_min_overlap"] <= 0.8:
                    raise ValueError("Tile overlap must be between 0.00 and 0.80")
                state.projects.save(project_directory, project)
                ui.notify("Project settings saved")
            except (CaptureValidationError, ValueError) as error:
                ui.notify(str(error), type="negative")

        ui.button("Save settings", on_click=save_settings)

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
                    ui.notify(
                        f"Prepared {prepared.panorama_count} panoramas and "
                        f"{prepared.cubemap_image_count} virtual camera images"
                    )
                    ui.navigate.to("/")
                except (CaptureValidationError, ValueError) as error:
                    ui.notify(str(error), type="negative")

            ui.button("Prepare 360 degree inputs", on_click=prepare_inputs).props("outline")

    render_page("Settings", content, current_path="/settings")
