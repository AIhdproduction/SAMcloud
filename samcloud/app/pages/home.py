"""Project dashboard page."""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from nicegui import ui

from samcloud.app.i18n import current_language, t, translate_or
from samcloud.app.dxf_time import dxf_ifc_created, dxf_ifc_seconds
from samcloud.app.layout import render_page
from samcloud.app.processing import (
    classification_engine,
    display_progress_text,
    reclassification_available,
    settings_use_flurseg,
)
from samcloud.app.state import state
from samcloud.share.ui import render_project_share_button


LOGGER = logging.getLogger(__name__)


_WORKFLOW = (
    ("Acquire imagery", "acquisition"),
    ("Align cameras", "alignment"),
    ("Add control points", "control_points"),
    ("Build dense cloud", "dense_cloud"),
    ("Classify", "classification"),
    ("Improve & filter", "refinement"),
    ("Export LAS / PLY", "export"),
    ("DXF/IFC export", "dxf"),
)

_PHASE_RANGES = {
    "acquisition": (0.0, 5.0),
    "alignment": (0.0, 57.0),
    "dense_cloud": (57.0, 84.0),
    "classification": (87.0, 93.0),
    "refinement": (93.0, 98.0),
}

_PHASE_STAGES = {
    "acquisition": ("prepare_images",),
    "alignment": ("feature_extraction", "matching", "mapping", "tiepoint_filtering"),
    "control_points": ("control_points",),
    "dense_cloud": ("undistortion", "dense_stereo", "dense_fusion"),
    "classification": ("classification",),
    "refinement": ("depth_prior", "point_cleaning", "depth_fill", "surface_fusion", "plane_fill",
                   "superpoint_vote", "geometry_refinement"),
}

# Order of the processing phases; the live status follows the estimator's running stage.
_PHASE_ORDER = ("acquisition", "alignment", "control_points", "dense_cloud", "classification", "refinement")

_TIME_GROUPS = (
    ("Bilder vorbereiten", ("Preparing images",)),
    ("Ausrichtung", ("Preparing GPS references", "Configuring panorama rig", "Feature extraction", "Image matching",
                     "Sparse reconstruction", "Tie point analysis", "GPS alignment")),
    ("Passpunkte", ("Preparing automatic control points", "Applying control points", "Preparing control points")),
    ("Dichte Punktwolke", ("Preparing dense reconstruction", "Dense stereo", "Dense fusion")),
    ("Klassifizierung", ("Preparing classification", "Preparing images for reclassification", "SAM3 classification",
                         "SAM3 classification (legacy log)", "Georeference")),
    ("Verbessern & filtern", ("Depth prior", "Mirror detection", "Point cleaning", "Depth fill", "Surface fusion",
                              "Plane filling", "Superpoint voting", "Geometric refinement")),
    ("Export", ("LAS export",)),
)

# Display keys of the processing-time groups; the German group names above stay the data keys.
_TIME_GROUP_KEYS = {
    "Bilder vorbereiten": "home.time_group.prepare_images",
    "Ausrichtung": "home.time_group.alignment",
    "Passpunkte": "home.time_group.control_points",
    "Dichte Punktwolke": "home.time_group.dense_cloud",
    "Klassifizierung": "home.time_group.classification",
    "Verbessern & filtern": "home.time_group.refinement",
    "Export": "home.time_group.export",
    "DXF/IFC-Export": "home.time_group.dxf_ifc",
    "Übrige Schritte": "home.time_group.other",
}

# Display keys of fixed progress stage texts written by the app (fallbacks, processing controller, estimator).
_STAGE_TEXT_KEYS = {
    "Processing": "home.stage.processing",
    "Preparing images": "home.stage.preparing_images",
    "Extracting features": "home.stage.extracting_features",
    "Matching images": "home.stage.matching_images",
    "Aligning cameras": "home.stage.aligning_cameras",
    "Waiting for control points": "home.stage.waiting_for_control_points",
    "Optimizing control points": "home.stage.optimizing_control_points",
    "Control points optimized": "home.stage.control_points_optimized",
    "Control-point optimization failed": "home.stage.control_point_optimization_failed",
    "Applying control points": "home.stage.applying_control_points",
    "Preparing dense reconstruction": "home.stage.preparing_dense_reconstruction",
    "Building dense cloud": "home.stage.building_dense_cloud",
    "Fusing dense cloud": "home.stage.fusing_dense_cloud",
    "Preparing classification": "home.stage.preparing_classification",
    "Classifying point cloud": "home.stage.classifying_point_cloud",
    "Reclassifying": "home.stage.reclassifying",
    "Improving and filtering": "home.stage.improving_and_filtering",
    "Exporting LAS": "home.stage.exporting_las",
    "Complete": "home.stage.complete",
    "Failed": "home.stage.failed",
    "Fertig": "home.stage.finished",
}


def _stage_text(snapshot: dict) -> str:
    """Display text of the current progress stage (estimator key first, then known fixed texts)."""
    stage = str(snapshot.get("stage") or "Processing")
    stage_key = snapshot.get("stage_key")
    fallback = str(display_progress_text(stage))
    if stage_key:
        return translate_or(f"workflow.stage.{stage_key}", fallback)
    key = _STAGE_TEXT_KEYS.get(stage)
    return translate_or(key, fallback) if key else fallback


def _status_text(status: str) -> str:
    """Display text of a workflow status ID (the ID itself stays unchanged in the data)."""
    key = str(status).replace(" ", "_")
    fallback = str(status).replace("_", " ")
    return translate_or(f"home.status.{key}", translate_or(f"workflow.status.{key}", fallback))


def _processing_time_summary(project_directory: Path | None) -> dict | None:
    """Total processing time of the latest completed run chain (alignment run + resume run)."""
    if project_directory is None:
        return None
    dxf_seconds = _dxf_ifc_seconds(project_directory)
    runs = []
    for path in (project_directory / "reports").glob("run_metrics_*.json"):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if isinstance(data.get("total_seconds"), (int, float)):
            runs.append(data)
    runs.sort(key=lambda item: str(item.get("started_at", "")))
    completed = [index for index, item in enumerate(runs) if item.get("status") == "Completed"]
    if not completed:
        return None
    chain = [runs[completed[-1]]]
    if "Sparse reconstruction" not in chain[0].get("stage_seconds", {}):
        for item in reversed(runs[:completed[-1]]):
            if "Sparse reconstruction" in item.get("stage_seconds", {}):
                chain.insert(0, item)
                break
    photogrammetry_total = float(sum(float(item["total_seconds"]) for item in chain))
    groups: dict[str, float] = {name: 0.0 for name, _ in _TIME_GROUPS}
    for item in chain:
        for stage, seconds in item.get("stage_seconds", {}).items():
            for name, members in _TIME_GROUPS:
                if stage in members or (name == "Ausrichtung" and (stage.startswith("Tie point filtering") or stage.startswith("Bundle adjustment"))):
                    groups[name] += float(seconds)
                    break
    other = max(0.0, photogrammetry_total - sum(groups.values()))
    if other >= 1.0:
        groups["Übrige Schritte"] = other
    if dxf_seconds is not None:
        groups["DXF/IFC-Export"] = dxf_seconds
    visible = {name: value for name, value in groups.items() if value >= 1.0}
    # The header and the Total line are the sum of exactly the lines shown, DXF/IFC included.
    total = photogrammetry_total + (dxf_seconds if dxf_seconds is not None and dxf_seconds >= 1.0 else 0.0)
    dxf_pending = dxf_seconds is None and not dxf_ifc_created(project_directory)
    return {"total_seconds": total, "groups": visible, "runs": len(chain), "dxf_pending": dxf_pending}


def _dxf_ifc_metrics(project_directory: Path | None) -> dict | None:
    """Timing record of the latest DXF/IFC run_all job (reports/dxf_run_metrics.json), if readable."""
    if project_directory is None:
        return None
    try:
        data = json.loads((project_directory / "reports" / "dxf_run_metrics.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def _dxf_ifc_seconds(project_directory: Path | None) -> float | None:
    """Time spent creating the DXF/IFC (indoor job metrics or drone environment steps), or None."""
    return dxf_ifc_seconds(project_directory)


def _dxf_ifc_step(project_directory: Path | None) -> dict:
    """Status of workflow step 8: {"status": ready|running|completed|failed, "percent": 0..100}."""
    if project_directory is None:
        return {"status": "ready", "percent": 0.0}
    job_file = project_directory / "products" / "dxf" / "job_status.json"
    try:
        job = json.loads(job_file.read_text(encoding="utf-8"))
        job_age = datetime.now().timestamp() - job_file.stat().st_mtime
    except (OSError, ValueError):
        job, job_age = {}, 0.0
    # A crashed job would leave "running" behind forever, so only a recently updated status counts.
    if isinstance(job, dict) and job.get("action") == "run_all" and job.get("state") == "running" and job_age < 1800:
        return {"status": "running", "percent": max(0.0, min(100.0, float(job.get("fraction") or 0.0) * 100.0))}
    metrics = _dxf_ifc_metrics(project_directory)
    if metrics is not None and metrics.get("status") == "failed":
        return {"status": "failed", "percent": 0.0}
    has_dxf = any((project_directory / "products" / "dxf" / "export").glob("*_clean.dxf"))
    has_ifc = any((project_directory / "products" / "bim").glob("*/*.ifc"))
    if has_dxf and has_ifc:
        return {"status": "completed", "percent": 100.0}
    return {"status": "ready", "percent": 0.0}


def _storage_summary(project_directory: Path | None) -> dict | None:
    """Load the completed project's storage totals for the workflow panel."""
    if project_directory is None:
        return None
    try:
        return json.loads((project_directory / "reports" / "storage_summary.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def render() -> None:
    """Render the active-project dashboard or a practical starting workspace."""
    project = state.active_project()
    title = t("home.title_empty") if project is None else project["project"]["name"].replace("-", " ").title()

    def content() -> None:
        with ui.row().classes("w-full gap-5 items-stretch").style("align-items: stretch"):
            with ui.column().classes("flex-1 gap-0").style("min-width: min(100%, 520px)"):
                _render_project_workspace(project)
            with ui.element("aside").classes("sam-surface sam-workflow").style("width: min(100%, 310px)"):
                _render_workflow(project)
        _render_recent_projects()

    render_page(title, content, current_path="/")


def _render_project_workspace(project: dict | None) -> None:
    """Render the useful entry action for either an empty or active workspace."""
    with ui.element("div").classes("sam-dropzone column items-center justify-center text-center"):
        ui.icon("photo_library")
        if project is None:
            ui.label(t("home.workspace.drop_title")).classes("sam-dropzone-title")
            ui.label(t("home.workspace.drop_hint")).classes("sam-muted text-sm q-mt-sm")
            with ui.row().classes("q-mt-xl gap-3"):
                ui.button(t("home.workspace.create_project"), icon="add", on_click=lambda: ui.navigate.to("/projects")).classes("sam-button-primary")
                ui.button(t("home.workspace.open_project"), icon="folder_open", on_click=lambda: ui.navigate.to("/projects")).props("outline").classes("sam-button-secondary")
            return

        acquisition = project.get("acquisition", {})
        sources = acquisition.get("sources", project.get("paths", {}))
        capture_type = acquisition.get("type", "drone_gps")
        source = (
            sources.get("panorama_image_directory") or sources.get("panorama_video_path")
            if capture_type == "panorama_360"
            else sources.get("image_directory", "")
        )
        ui.icon("folder_open")
        ui.label(t("home.workspace.ready")).classes("sam-dropzone-title")
        ui.label(source or t("home.workspace.no_source")).classes("sam-muted text-sm q-mt-sm").style("max-width: 80%; overflow-wrap: anywhere")
        workflow = _live_workflow(project)
        with ui.row().classes("q-mt-xl gap-3"):
            ui.button(t("home.workspace.review_settings"), icon="tune", on_click=lambda: ui.navigate.to("/settings")).classes("sam-button-primary")
            sparse_ready = workflow.get("alignment") == "completed"
            sparse_button = ui.button(
                t("home.workspace.open_sparse"),
                icon="hub",
                on_click=lambda: ui.navigate.to("/sparse"),
            ).props("outline").classes("sam-button-secondary")
            sparse_button.set_enabled(sparse_ready)
        processing_active = state.processing.is_running(state.active_project_directory) if state.active_project_directory else False
        mode = workflow.get("control_points_mode", "none")
        if processing_active:
            ui.label(t("home.workspace.processing_running")).classes("sam-muted text-sm q-mt-md")
        if processing_active or _workflow_is_active(workflow):
            _render_processing_progress(project)
        elif workflow.get("control_points") == "waiting":
            ui.button(
                t("home.workspace.continue_processing"),
                icon="play_arrow",
                on_click=lambda: _continue_after_control_points(),
            ).classes("sam-button-primary q-mt-md")
            ui.button(
                t("home.workspace.review_control_points"),
                icon="my_location",
                on_click=lambda: ui.navigate.to("/control-points"),
            ).props("outline").classes("sam-button-secondary q-mt-md q-ml-sm")
        elif workflow.get("classification") != "completed":
            action_label = (
                t("home.workspace.start_after_alignment")
                if mode == "after_alignment"
                else t("home.workspace.start_automatic")
            )
            ui.button(action_label, icon="play_arrow", on_click=lambda: _start_processing()).classes("sam-button-primary q-mt-md")
        elif workflow.get("export") != "completed":
            ui.button(t("home.workspace.export_las"), icon="file_download", on_click=lambda: _export_las()).classes("sam-button-primary q-mt-md")
        if not processing_active and not _workflow_is_active(workflow) and _cache_available():
            ui.button(
                t("home.workspace.refine_from_cache"),
                icon="refresh",
                on_click=lambda: _refine_from_cache(),
            ).props("outline").classes("sam-button-secondary q-mt-md").tooltip(
                t("home.workspace.refine_from_cache_tooltip")
            )
        if (not processing_active and not _workflow_is_active(workflow)
                and workflow.get("classification") in {"completed", "failed"} and _reclassification_available()):
            engine_name = "FlurSeg" if classification_engine(project.get("settings", {}).get("classification", {})) == "flurseg" else "SAM3"
            ui.button(
                t("home.workspace.reclassify", engine=engine_name),
                icon="auto_awesome",
                on_click=lambda: _reclassify(),
            ).props("outline").classes("sam-button-secondary q-mt-md q-ml-sm").tooltip(
                t("home.workspace.reclassify_tooltip")
            )
        if project.get("processing", {}).get("last_error"):
            ui.label(str(display_progress_text(project["processing"]["last_error"]))).classes("text-red-4 text-sm q-mt-md")


def _dxf_unavailable_reason(project: dict | None) -> str | None:
    """Return why the DXF step cannot be opened yet, or None when it can (file existence checks only)."""
    from samcloud.app.dxf_controller import dxf_unavailable_reason

    return dxf_unavailable_reason(state.active_project_directory if project is not None else None)


def _workflow_steps(project: dict | None) -> tuple[tuple[str, str], ...]:
    """Workflow steps shown in the side panel; FlurSeg projects have no DXF/IFC step."""
    if project is not None and settings_use_flurseg(project.get("settings", {})):
        return tuple(step for step in _WORKFLOW if step[1] != "dxf")
    return _WORKFLOW


def _workflow_is_active(workflow: dict) -> bool:
    """Return whether any persisted workflow stage still needs processing."""
    return any(
        status in {"running", "queued"}
        for status in workflow.values()
        if isinstance(status, str)
    )


def _auto_without_control_points(project: dict | None, project_directory: Path | None = None) -> bool:
    """Whether the project is in ``auto`` mode but has no ``control_points.csv`` (so the GCP step does not apply)."""
    if project is None:
        return False
    if project.get("workflow", {}).get("control_points_mode", "auto") != "auto":
        return False
    directory = project_directory if project_directory is not None else state.active_project_directory
    if directory is None:
        return True
    return not (Path(directory) / "control_points" / "control_points.csv").is_file()


def _control_points_status_text(project: dict | None, status: str) -> str:
    """Status text of the control-point step; auto mode without control points reads 'skipped (no control points)'."""
    if status in {"skipped", "not_started", "not started"} and _auto_without_control_points(project):
        return t("home.status.skipped_no_control_points")
    return _status_text(status)


def _live_workflow(project: dict | None, snapshot: dict | None = None) -> dict:
    """Overlay live pipeline progress onto persisted workflow statuses, then skip the idle GCP step in auto mode.

    ``snapshot`` lets a caller pass the progress it already read, so statuses and phase texts
    come from the same snapshot.
    """
    statuses = _overlay_live_progress(project, snapshot)
    if (
        project is not None
        and project.get("workflow", {}).get("control_points") in {"not_started", "skipped"}
        and statuses.get("control_points") in {"not_started", "completed"}
        and _auto_without_control_points(project)
    ):
        statuses["control_points"] = "skipped"
    return statuses


def _overlay_live_progress(project: dict | None, snapshot: dict | None = None) -> dict:
    """Overlay live pipeline progress onto persisted workflow statuses."""
    if project is None:
        return {}
    statuses = dict(project.get("workflow", {}))
    if not _workflow_is_active(statuses):
        return statuses
    if snapshot is None:
        snapshot = _read_progress(project, state.active_project_directory)
    percent = float(snapshot.get("percent", 0))
    stage = str(snapshot.get("stage", ""))
    if stage == "Waiting for control points":
        statuses.update({
            "alignment": "completed",
            "control_points": "waiting",
            "dense_cloud": "not_started",
            "classification": "not_started",
        })
    else:
        running_phase = _phase_of_stage(snapshot.get("stage_key"), snapshot)
        if running_phase is not None:
            position = _PHASE_ORDER.index(running_phase)
            has_prepare = any(item.get("key") == "prepare_images" for item in snapshot.get("stages", []))
            for phase in _PHASE_ORDER[:position]:
                if phase == "acquisition" and not has_prepare:
                    continue  # resume runs and older progress files: step 1 stays "ready"
                if statuses.get(phase) not in {"skipped", "waiting"}:
                    statuses[phase] = "completed"
            statuses[running_phase] = "running"
            return statuses
        if percent >= 57:
            statuses["alignment"] = "completed"
            statuses["dense_cloud"] = "completed" if percent >= 87 else "running"
        if percent >= 87:
            statuses["classification"] = "completed" if percent >= 93 else "running"
        if percent >= 93:
            statuses["refinement"] = "completed" if percent >= 100 else "running"
    return statuses


def _has_prepare_stage(progress: dict) -> bool:
    """Whether the run has an image-preparation stage (only then step 1 gets a bar)."""
    return any(item.get("key") == "prepare_images" for item in progress.get("stages", []))


def _phase_of_stage(stage_key, progress: dict | None = None) -> str | None:
    """Workflow phase that contains an estimator stage key.

    With ``progress``, a key without a fixed phase (e.g. a stage a newer pipeline plans) joins the
    phase of the planned stage before it, so every remaining second belongs to exactly one phase.
    """
    for phase, members in _PHASE_STAGES.items():
        if stage_key in members:
            return phase
    if progress is not None:
        return _stage_phases(progress).get(stage_key)
    return None


def _stage_phases(progress: dict) -> dict:
    """Phase of every stage in an estimator snapshot (unknown keys follow the stage before them)."""
    phases: dict = {}
    previous = None
    for item in progress.get("stages") or []:
        if not isinstance(item, dict):
            continue
        key = item.get("key")
        phase = next((name for name, members in _PHASE_STAGES.items() if key in members), None) or previous
        if phase is not None:
            phases[key] = phase
            previous = phase
    return phases


def _phase_items(stage: str, progress: dict) -> list[dict]:
    """Estimator stages of one workflow phase, in plan order."""
    phases = _stage_phases(progress)
    return [item for item in progress.get("stages") or []
            if isinstance(item, dict) and phases.get(item.get("key")) == stage]


def _phase_eta(stage: str, statuses: dict, progress: dict) -> float | None:
    """Remaining seconds of one workflow phase from the estimator snapshot.

    Uses the same numbers as the snapshot's ``eta_seconds`` (running stage: ``stage_eta_seconds``,
    pending stages: ``expected_seconds``), so the phases' remaining times add up to the total ETA.
    """
    if statuses.get(stage) in {"completed", "skipped"}:
        return None
    remaining = 0.0
    found = False
    for item in _phase_items(stage, progress):
        if item.get("status") in {"done", "skipped"}:
            continue
        found = True
        if item.get("status") == "running" and progress.get("stage_eta_seconds") is not None:
            remaining += float(progress["stage_eta_seconds"])
        else:
            remaining += float(item.get("expected_seconds") or 0.0)
    return remaining if found else None


def _phase_text(stage: str, statuses: dict, progress: dict) -> str:
    percent = _phase_percent(stage, statuses, progress)
    # A remaining time only belongs to a phase of a run that is actually going on.
    eta = _phase_eta(stage, statuses, progress) if statuses.get(stage) in {"running", "queued"} else None
    detail = ""
    if stage == "acquisition" and statuses.get(stage) == "running" and progress.get("stage_key") == "prepare_images":
        detail = str(display_progress_text(progress.get("detail") or ""))
    parts = [f"{percent:.0f}%"]
    if detail:
        parts.append(detail)
    if eta is not None and percent < 100.0:
        parts.append(_format_span(eta))
    return " · ".join(parts)


# progress.json snapshots by path: (file signature, parsed content).
_PROGRESS_FILE_CACHE: dict[str, tuple[tuple[int, int], dict]] = {}


def _load_progress_file(path: Path) -> dict | None:
    """Parsed progress.json; every reader of an unchanged file gets the same snapshot.

    The main progress bar and the workflow panel refresh on separate timers; reading through this
    cache keeps their stage and total ETAs from one snapshot. A half-written file keeps the last one.
    """
    key = str(path)
    try:
        stat = path.stat()
    except OSError:
        return None
    signature = (stat.st_mtime_ns, stat.st_size)
    cached = _PROGRESS_FILE_CACHE.get(key)
    if cached is not None and cached[0] == signature:
        return dict(cached[1])
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return dict(cached[1]) if cached is not None else None
    if not isinstance(data, dict):
        return None
    _PROGRESS_FILE_CACHE[key] = (signature, data)
    return dict(data)


def _read_progress(project: dict, project_directory: Path | None) -> dict:
    """Read pipeline progress, with a stage-based fallback for older runs."""
    processing = project.get("processing", {})
    progress = dict(processing.get("progress", {}))
    if project_directory is not None:
        progress.update(_load_progress_file(project_directory / "reports" / "progress.json") or {})

    workflow = project.get("workflow", {})
    if project_directory is not None and not progress.get("stage"):
        log_path = project_directory / "reports" / "pipeline.log"
        if log_path.exists():
            try:
                log_lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
                last_command = next(
                    (line.lower() for line in reversed(log_lines) if line.startswith(">>>")),
                    "",
                )
                log_stage = (
                    ("patch_match_stereo", 62, "Building dense cloud"),
                    ("stereo_fusion", 78, "Fusing dense cloud"),
                    ("model_converter", 84, "Preparing classification"),
                    ("classify.py", 87, "Classifying point cloud"),
                    ("image_undistorter", 57, "Preparing dense reconstruction"),
                    ("mapper", 32, "Aligning cameras"),
                    ("feature_extractor", 5, "Extracting features"),
                    ("matcher", 18, "Matching images"),
                )
                for marker, percent, stage in log_stage:
                    if marker in last_command:
                        progress.update({"percent": percent, "stage": stage})
                        break
            except OSError:
                pass
    fallback = (
        ("alignment", 32, "Aligning cameras"),
        ("dense_cloud", 62, "Building dense cloud"),
        ("classification", 87, "Classifying point cloud"),
        ("export", 98, "Exporting LAS"),
    )
    if not progress.get("stage"):
        for key, percent, stage in fallback:
            if workflow.get(key) in {"running", "queued"}:
                progress.update({"percent": percent, "stage": stage})
                break

    if not progress:
        progress = {"percent": 0, "stage": "Preparing images", "detail": ""}
    return progress


def _format_span(seconds: float | None) -> str:
    """Duration as "1h 05m" / "2m 05s" (or the "calculating" text when unknown)."""
    if seconds is None or seconds < 0:
        return t("home.duration.calculating")
    total_seconds = int(seconds)
    minutes, remaining_seconds = divmod(total_seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return t("home.duration.hours_minutes", hours=hours, minutes=f"{minutes:02d}")
    return t("home.duration.minutes_seconds", minutes=minutes, seconds=f"{remaining_seconds:02d}")


def _format_duration(seconds: float | None) -> str:
    if seconds is None or seconds < 0:
        return _format_span(seconds)
    return t("common.remaining", time=_format_span(seconds))


def _format_elapsed(seconds: float | None) -> str:
    return _format_span(seconds)


def _render_processing_progress(project: dict) -> None:
    """Render a live progress bar and refresh it while processing runs."""
    initial = _read_progress(project, state.active_project_directory)
    progress_bar = ui.linear_progress(
        value=float(initial.get("percent", 0)) / 100,
        show_value=False,
    ).classes("w-full q-mt-md")
    stage_label = ui.label().classes("sam-muted text-sm")
    photo_label = ui.label().classes("sam-muted text-sm")
    eta_label = ui.label().classes("sam-muted text-sm")

    def refresh() -> None:
        current_project = state.active_project()
        if current_project is None:
            return
        snapshot = _read_progress(current_project, state.active_project_directory)
        percent = max(0.0, min(100.0, float(snapshot.get("percent", 0))))
        started_at = (
            snapshot.get("started_at")
            or current_project.get("processing", {}).get("started_at")
            or current_project.get("project", {}).get("updated_at")
        )
        elapsed = None
        if isinstance(started_at, (int, float)):
            elapsed = max(0.0, datetime.now(timezone.utc).timestamp() - float(started_at))
        elif isinstance(started_at, str):
            try:
                elapsed = max(0.0, (datetime.now(timezone.utc) - datetime.fromisoformat(started_at)).total_seconds())
            except ValueError:
                pass
        eta = snapshot.get("eta_seconds")
        if eta is None:
            eta = elapsed * (100.0 - percent) / percent if elapsed and percent > 0 else None
        progress_bar.set_value(percent / 100.0)
        stage_text = _stage_text(snapshot)
        if snapshot.get("stage_eta_seconds") is not None:
            stage_label.set_text(t(
                "home.progress.stage_with_eta",
                stage=stage_text,
                percent=f"{float(snapshot.get('stage_percent', 0)):.0f}",
                time=_format_span(snapshot["stage_eta_seconds"]),
            ))
        else:
            stage_label.set_text(t("home.progress.stage_percent", stage=stage_text, percent=f"{percent:.0f}"))
        if snapshot.get("stage_key") == "prepare_images" and snapshot.get("detail"):
            stage_label.set_text(f"{stage_label.text} · {display_progress_text(snapshot['detail'])}")
        if snapshot.get("photo_total"):
            photo_label.set_text(t(
                "home.progress.photos",
                processed=snapshot["photo_processed"],
                total=snapshot["photo_total"],
                percent=f"{snapshot['photo_percent']:.0f}",
            ))
        else:
            photo_label.set_text("")
        eta_label.set_text(t("home.progress.elapsed_eta", elapsed=_format_elapsed(elapsed), eta=_format_duration(eta)))

    refresh()
    ui.timer(1.0, refresh)


def _render_workflow(project: dict | None) -> None:
    ui.add_head_html(
        "<style>"
        ".sam-processing-time { margin-top:14px; padding-top:12px; border-top:1px solid var(--sam-line); }"
        ".sam-processing-total { font-size:15px; font-weight:600; color:var(--sam-text); margin-bottom:6px; }"
        "</style>"
    )
    ui.label(t("home.workflow.title")).classes("sam-workflow-title")
    progress = _read_progress(project, state.active_project_directory) if project else {}
    statuses = _live_workflow(project, progress)
    phase_bars = {}
    phase_labels = {}
    status_labels = {}
    dxf_reason = _dxf_unavailable_reason(project)
    dxf_step = _dxf_ifc_step(state.active_project_directory) if project is not None else {"status": "ready", "percent": 0.0}
    dxf_bar = None
    dxf_status_label = None
    for index, (label, key) in enumerate(_workflow_steps(project), start=1):
        if key == "dxf" and project is not None:
            status = dxf_step["status"] if dxf_step["status"] != "ready" else ("ready" if dxf_reason is None else dxf_reason)
        else:
            status = (
                statuses.get(key, "ready") if key == "acquisition" and project is not None
                else "ready" if key is None and project is not None
                else statuses.get(key, "not started")
            )
        step_class = "sam-workflow-step"
        if index == 1 and project is None:
            step_class += " is-active"
        elif status == "completed":
            step_class += " is-complete"
        elif status == "running" or (status == "ready" and key != "dxf"):
            step_class += " is-active"
        with ui.element("div").classes(step_class):
            ui.label(str(index)).classes("sam-step-number")
            ui.label(translate_or(f"workflow.step.{key}", label)).classes("sam-step-label")
            if key == "dxf" and project is not None and dxf_step["status"] == "ready" and dxf_reason is not None:
                status_label = ui.label(status.replace("_", " ")).classes("sam-step-status")
                status_label.style("text-transform:none").mark("home-dxf-status")
            else:
                status_text = _control_points_status_text(project, status) if key == "control_points" else _status_text(status)
                status_label = ui.label(status_text).classes("sam-step-status")
                if current_language() != "en":
                    # Translated statuses are already cased correctly; CSS capitalize would capitalize every word.
                    status_label.style("text-transform:none")
                status_labels[key] = status_label
                if key == "dxf":
                    status_label.mark("home-dxf-step-status")
                    dxf_status_label = status_label
            if key == "dxf" and project is not None:
                if dxf_reason is None:
                    ui.button(t("home.workflow.open_dxf"), icon="architecture", on_click=lambda: ui.navigate.to("/dxf")).props("flat dense no-caps").classes("text-cyan-4 text-xs q-mt-xs").mark("home-open-dxf")
                else:
                    # A disabled q-btn swallows hover events, so the tooltip lives on a wrapping element.
                    with ui.element("div").classes("inline-block q-mt-xs") as dxf_slot:
                        ui.button(t("home.workflow.open_dxf"), icon="architecture").props("flat dense no-caps disable").classes("text-xs").mark("home-open-dxf")
                    dxf_slot.tooltip(dxf_reason)
                if dxf_step["status"] in {"completed", "running"}:
                    with ui.row().classes("items-center gap-2 w-full q-mt-xs"):
                        dxf_bar = ui.linear_progress(value=dxf_step["percent"] / 100.0, show_value=False).classes("flex-1").mark("home-dxf-bar")
            show_bar = key in _PHASE_RANGES and project is not None
            if key == "acquisition" and show_bar:
                show_bar = _has_prepare_stage(progress)
            if show_bar:
                phase_percent = _phase_percent(key, statuses, progress)
                with ui.row().classes("items-center gap-2 w-full q-mt-xs"):
                    phase_bar = ui.linear_progress(
                        value=phase_percent / 100.0,
                        show_value=False,
                    ).classes("flex-1")
                    phase_label = ui.label(_phase_text(key, statuses, progress)).classes("text-xs sam-muted")
                phase_bars[key] = phase_bar
                phase_labels[key] = phase_label

    if project is not None and project.get("workflow", {}).get("classification") == "completed":
        timing = _processing_time_summary(state.active_project_directory)
        if timing is not None:
            with ui.element("div").classes("sam-processing-time"):
                ui.label(t("home.workflow.total_time", time=_format_elapsed(timing["total_seconds"]))).classes("sam-processing-total")
                for name, seconds in timing["groups"].items():
                    with ui.row().classes("items-center justify-between w-full no-wrap"):
                        group_key = _TIME_GROUP_KEYS.get(name)
                        ui.label(translate_or(group_key, name) if group_key else name).classes("text-xs sam-muted")
                        ui.label(_format_elapsed(seconds)).classes("text-xs sam-muted")
                if timing.get("dxf_pending"):
                    with ui.row().classes("items-center justify-between w-full no-wrap"):
                        ui.label(translate_or("home.time_group.dxf_ifc", "DXF/IFC")).classes("text-xs sam-muted")
                        ui.label(t("home.workflow.not_created")).classes("text-xs sam-muted")
                with ui.row().classes("items-center justify-between w-full no-wrap q-mt-xs"):
                    ui.label(t("home.workflow.total")).classes("text-xs").style("font-weight:600")
                    ui.label(_format_elapsed(timing["total_seconds"])).classes("text-xs").style("font-weight:600").mark("home-total-time")
        storage = _storage_summary(state.active_project_directory)
        if storage is not None:
            after = float(storage.get("after_bytes", 0)) / 1024 ** 3
            ui.label(t("home.workflow.storage", size=f"{after:.1f}")).classes("text-xs sam-muted")

    project_share_slot = None
    previous_dxf_status = dxf_step["status"]
    previous_classification_status = statuses.get("classification")
    if project is not None:
        with ui.column().classes("w-full") as project_share_slot:
            def render_project_share(current_project: dict) -> None:
                try:
                    project_share_slot.clear()
                    with project_share_slot:
                        render_project_share_button(state.active_project_directory, current_project)
                except Exception as error:
                    LOGGER.warning("project share button unavailable: %s", error, exc_info=True)

            render_project_share(project)

    if project is not None and dxf_step["status"] == "running" and dxf_bar is not None and dxf_status_label is not None:
        def refresh_dxf_step() -> None:
            nonlocal previous_dxf_status
            current = _dxf_ifc_step(state.active_project_directory)
            dxf_bar.set_value(current["percent"] / 100.0)
            dxf_status_label.set_text(_status_text(current["status"]))
            if current["status"] == "completed" and previous_dxf_status != "completed" and project_share_slot is not None:
                current_project = state.active_project()
                if current_project is not None:
                    render_project_share(current_project)
            previous_dxf_status = current["status"]

        ui.timer(2.0, refresh_dxf_step)

    if project is not None and _workflow_is_active(project.get("workflow", {})):
        def refresh_workflow() -> None:
            nonlocal previous_classification_status, previous_dxf_status
            current_project = state.active_project()
            if current_project is None:
                return
            current_progress = _read_progress(current_project, state.active_project_directory)
            current_statuses = _live_workflow(current_project, current_progress)
            for stage, bar in phase_bars.items():
                value = _phase_percent(stage, current_statuses, current_progress)
                bar.set_value(value / 100.0)
                phase_labels[stage].set_text(_phase_text(stage, current_statuses, current_progress))
            for stage, label in status_labels.items():
                if stage in current_statuses:
                    value = str(current_statuses[stage])
                    label.set_text(
                        _control_points_status_text(current_project, value) if stage == "control_points" else _status_text(value)
                    )
            current_dxf_step = _dxf_ifc_step(state.active_project_directory)
            project_share_became_available = (
                current_dxf_step["status"] == "completed" and previous_dxf_status != "completed"
            ) or (
                current_statuses.get("classification") == "completed"
                and previous_classification_status != "completed"
            )
            if project_share_became_available and project_share_slot is not None:
                render_project_share(current_project)
            previous_dxf_status = current_dxf_step["status"]
            previous_classification_status = current_statuses.get("classification")

        ui.timer(1.0, refresh_workflow)


def _phase_percent(stage: str, statuses: dict, progress: dict) -> float:
    """Convert global pipeline progress into an independent phase percentage."""
    if statuses.get(stage) == "completed":
        return 100.0
    estimator_stages = _phase_items(stage, progress)
    if estimator_stages:
        # A step that has not started (only pending/skipped stages) stays at 0 %: skipped stages
        # only count as complete once the step runs, so a queued step never shows progress.
        started = any(
            item.get("status") in {"running", "done"}
            or (item.get("status") != "skipped" and float(item.get("percent") or 0.0) > 0.0)
            for item in estimator_stages
        )
        has_pending = any(item.get("status") not in {"done", "skipped"} for item in estimator_stages)
        if not started and has_pending:
            return 0.0
        weights = [max(float(item.get("expected_seconds") or 0.0), 1.0) for item in estimator_stages]
        values = [100.0 if item.get("status") in {"done", "skipped"} else float(item.get("percent") or 0.0) for item in estimator_stages]
        return sum(value * weight for value, weight in zip(values, weights)) / sum(weights)
    if stage not in _PHASE_RANGES:
        return 0.0
    start, end = _PHASE_RANGES[stage]
    if statuses.get(stage) == "completed" or float(progress.get("percent", 0)) >= end:
        return 100.0
    if statuses.get(stage) in {"not_started", "queued"} and float(progress.get("percent", 0)) <= start:
        return 0.0
    global_percent = max(start, min(end, float(progress.get("percent", 0))))
    return (global_percent - start) / (end - start) * 100.0


def _render_recent_projects() -> None:
    from samcloud.app.pages.projects import format_created

    projects = state.projects.discover_projects()
    with ui.element("section").classes("sam-surface sam-recent"):
        with ui.row().classes("sam-recent-head items-center justify-between w-full"):
            ui.label(t("home.recent.title")).classes("sam-section-title")
            ui.button(t("home.recent.view_all"), icon="arrow_forward", on_click=lambda: ui.navigate.to("/projects")).props("flat no-caps").classes("text-cyan-4")
        if not projects:
            ui.label(t("home.recent.empty")).classes("sam-empty-row")
            return
        for project_directory in projects[:5]:
            project = state.projects.load(project_directory)
            with ui.row().classes("sam-project-row w-full items-center no-wrap"):
                ui.icon("folder").classes("text-cyan-4 q-mr-md")
                with ui.column().classes("gap-0 flex-1").style("min-width: 0"):
                    ui.label(project["project"]["name"]).classes("sam-project-name")
                    ui.label(str(project_directory)).classes("sam-project-path")
                created_text = format_created(project, project_directory, current_language())
                if created_text:
                    ui.label(created_text).classes("text-xs text-slate-500 whitespace-nowrap")
                ui.button(t("common.open"), on_click=lambda path=project_directory: _open_project(path)).props("flat no-caps").classes("text-cyan-4")


def _open_project(project_directory: Path) -> None:
    state.open_project(project_directory)
    ui.navigate.to("/")


def _start_processing() -> None:
    if state.active_project_directory is None:
        return
    try:
        state.processing.start_pipeline(state.active_project_directory)
        ui.notify(t("home.notify.processing_started"))
        ui.navigate.to("/")
    except (OSError, ValueError) as error:
        ui.notify(str(error), type="negative")


def _continue_after_control_points() -> None:
    if state.active_project_directory is None:
        return
    try:
        state.processing.continue_after_control_points(state.active_project_directory)
        ui.notify(t("home.notify.control_points_started"))
        ui.navigate.to("/")
    except (OSError, ValueError) as error:
        ui.notify(str(error), type="negative")


def _cache_available() -> bool:
    from samcloud.engine.processing_cache import load_manifest

    return state.active_project_directory is not None and load_manifest(state.active_project_directory) is not None


def _reclassification_available() -> bool:
    return state.active_project_directory is not None and reclassification_available(state.active_project_directory)


def _reclassify() -> None:
    if state.active_project_directory is None:
        return
    try:
        state.processing.reclassify(state.active_project_directory)
        ui.notify(t("home.notify.reclassify_started"))
        ui.navigate.to("/")
    except (OSError, ValueError) as error:
        ui.notify(str(error), type="negative")


def _refine_from_cache() -> None:
    if state.active_project_directory is None:
        return
    try:
        state.processing.refine_from_cache(state.active_project_directory)
        ui.notify(t("home.notify.refine_started"))
        ui.navigate.to("/")
    except (OSError, ValueError) as error:
        ui.notify(str(error), type="negative")


def _export_las() -> None:
    if state.active_project_directory is None:
        return
    try:
        state.processing.export_las(state.active_project_directory)
        ui.notify(t("home.notify.las_export_started"))
        ui.navigate.to("/")
    except (OSError, ValueError) as error:
        ui.notify(str(error), type="negative")
