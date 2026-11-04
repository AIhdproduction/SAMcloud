"""Project dashboard page."""

import json
from datetime import datetime, timezone
from pathlib import Path

from nicegui import ui

from samcloud.app.layout import render_page
from samcloud.app.state import state


_WORKFLOW = (
    ("Acquire imagery", None),
    ("Align cameras", "alignment"),
    ("Add control points", "control_points"),
    ("Build dense cloud", "dense_cloud"),
    ("Classify with SAM3", "classification"),
    ("Export LAS / PLY", "export"),
)

_PHASE_RANGES = {
    "alignment": (0.0, 57.0),
    "dense_cloud": (57.0, 84.0),
    "classification": (87.0, 98.0),
}

_PHASE_STAGES = {
    "alignment": ("feature_extraction", "matching", "mapping", "tiepoint_filtering"),
    "control_points": ("control_points",),
    "dense_cloud": ("undistortion", "dense_stereo", "dense_fusion"),
    "classification": ("classification", "geometry_refinement"),
}

_TIME_GROUPS = (
    ("Ausrichtung", ("Preparing GPS references", "Feature extraction", "Image matching", "Sparse reconstruction",
                     "Tie point analysis", "GPS alignment")),
    ("Passpunkte", ("Preparing automatic control points", "Applying control points", "Preparing control points")),
    ("Dichte Punktwolke", ("Preparing dense reconstruction", "Dense stereo", "Dense fusion")),
    ("Klassifizierung", ("Preparing classification", "SAM3 classification", "Geometric refinement", "Georeference")),
    ("Export", ("LAS export",)),
)


def _processing_time_summary(project_directory: Path | None) -> dict | None:
    """Total processing time of the latest completed run chain (alignment run + resume run)."""
    if project_directory is None:
        return None
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
    total = float(sum(float(item["total_seconds"]) for item in chain))
    groups: dict[str, float] = {name: 0.0 for name, _ in _TIME_GROUPS}
    for item in chain:
        for stage, seconds in item.get("stage_seconds", {}).items():
            for name, members in _TIME_GROUPS:
                if stage in members or (name == "Ausrichtung" and (stage.startswith("Tie point filtering") or stage.startswith("Bundle adjustment"))):
                    groups[name] += float(seconds)
                    break
    other = max(0.0, total - sum(groups.values()))
    if other >= 1.0:
        groups["Übrige Schritte"] = other
    return {"total_seconds": total, "groups": {name: value for name, value in groups.items() if value >= 1.0}, "runs": len(chain)}


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
    title = "Ready for your next reconstruction?" if project is None else project["project"]["name"].replace("-", " ").title()

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
            ui.label("Drop drone images or a 360° source here").classes("sam-dropzone-title")
            ui.label("Create a project first, then select the image directory or panorama source.").classes("sam-muted text-sm q-mt-sm")
            with ui.row().classes("q-mt-xl gap-3"):
                ui.button("Create project", icon="add", on_click=lambda: ui.navigate.to("/projects")).classes("sam-button-primary")
                ui.button("Open project", icon="folder_open", on_click=lambda: ui.navigate.to("/projects")).props("outline").classes("sam-button-secondary")
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
        ui.label("Project workspace is ready").classes("sam-dropzone-title")
        ui.label(source or "No image source selected yet").classes("sam-muted text-sm q-mt-sm").style("max-width: 80%; overflow-wrap: anywhere")
        workflow = _live_workflow(project)
        with ui.row().classes("q-mt-xl gap-3"):
            ui.button("Review project settings", icon="tune", on_click=lambda: ui.navigate.to("/settings")).classes("sam-button-primary")
            sparse_ready = workflow.get("alignment") == "completed"
            sparse_button = ui.button(
                "Open sparse cloud",
                icon="hub",
                on_click=lambda: ui.navigate.to("/sparse"),
            ).props("outline").classes("sam-button-secondary")
            sparse_button.set_enabled(sparse_ready)
        processing_active = state.processing.is_running(state.active_project_directory) if state.active_project_directory else False
        mode = workflow.get("control_points_mode", "none")
        if processing_active:
            ui.label("Processing is running. The status is saved in the project.").classes("sam-muted text-sm q-mt-md")
        if processing_active or _workflow_is_active(workflow):
            _render_processing_progress(project)
        elif workflow.get("control_points") == "waiting":
            ui.button(
                "Continue processing",
                icon="play_arrow",
                on_click=lambda: _continue_after_control_points(),
            ).classes("sam-button-primary q-mt-md")
            ui.button(
                "Review control points",
                icon="my_location",
                on_click=lambda: ui.navigate.to("/control-points"),
            ).props("outline").classes("sam-button-secondary q-mt-md q-ml-sm")
        elif workflow.get("classification") != "completed":
            action_label = (
                "Start alignment and wait for control points"
                if mode == "after_alignment"
                else "Start automatic processing"
            )
            ui.button(action_label, icon="play_arrow", on_click=lambda: _start_processing()).classes("sam-button-primary q-mt-md")
        elif workflow.get("export") != "completed":
            ui.button("Export LAS manually", icon="file_download", on_click=lambda: _export_las()).classes("sam-button-primary q-mt-md")
        if project.get("processing", {}).get("last_error"):
            ui.label(project["processing"]["last_error"]).classes("text-red-4 text-sm q-mt-md")


def _workflow_is_active(workflow: dict) -> bool:
    """Return whether any persisted workflow stage still needs processing."""
    return any(
        status in {"running", "queued"}
        for status in workflow.values()
        if isinstance(status, str)
    )


def _live_workflow(project: dict | None) -> dict:
    """Overlay live pipeline progress onto persisted workflow statuses."""
    if project is None:
        return {}
    statuses = dict(project.get("workflow", {}))
    if not _workflow_is_active(statuses):
        return statuses
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
        if percent >= 57:
            statuses["alignment"] = "completed"
            statuses["dense_cloud"] = "completed" if percent >= 87 else "running"
        if percent >= 87:
            statuses["classification"] = "completed" if percent >= 100 else "running"
    return statuses


def _read_progress(project: dict, project_directory: Path | None) -> dict:
    """Read pipeline progress, with a stage-based fallback for older runs."""
    processing = project.get("processing", {})
    progress = dict(processing.get("progress", {}))
    if project_directory is not None:
        progress_path = project_directory / "reports" / "progress.json"
        if progress_path.exists():
            try:
                progress.update(json.loads(progress_path.read_text(encoding="utf-8")))
            except (OSError, ValueError):
                pass

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


def _format_duration(seconds: float | None) -> str:
    if seconds is None or seconds < 0:
        return "calculating..."
    total_seconds = int(seconds)
    minutes, remaining_seconds = divmod(total_seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h {minutes:02d}m remaining"
    return f"{minutes}m {remaining_seconds:02d}s remaining"


def _format_elapsed(seconds: float | None) -> str:
    formatted = _format_duration(seconds)
    return formatted.removesuffix(" remaining")


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
        if snapshot.get("stage_eta_seconds") is not None:
            stage_label.set_text(
                f"{snapshot.get('stage', 'Processing')} · {float(snapshot.get('stage_percent', 0)):.0f}% · "
                f"Schritt noch {_format_duration(snapshot['stage_eta_seconds'])}"
            )
        else:
            stage_label.set_text(f"{snapshot.get('stage', 'Processing')} · {percent:.0f}%")
        if snapshot.get("photo_total"):
            photo_label.set_text(
                f"Photos: {snapshot['photo_processed']} / {snapshot['photo_total']} "
                f"({snapshot['photo_percent']:.0f}%)"
            )
        else:
            photo_label.set_text("")
        eta_label.set_text(f"Elapsed: {_format_elapsed(elapsed)} · Gesamt-ETA: {_format_duration(eta)}")

    refresh()
    ui.timer(1.0, refresh)


def _render_workflow(project: dict | None) -> None:
    ui.add_head_html(
        "<style>"
        ".sam-processing-time { margin-top:14px; padding-top:12px; border-top:1px solid var(--sam-line); }"
        ".sam-processing-total { font-size:15px; font-weight:600; color:var(--sam-text); margin-bottom:6px; }"
        "</style>"
    )
    ui.label("Project workflow").classes("sam-workflow-title")
    statuses = _live_workflow(project)
    progress = _read_progress(project, state.active_project_directory) if project else {}
    phase_bars = {}
    phase_labels = {}
    status_labels = {}
    for index, (label, key) in enumerate(_WORKFLOW, start=1):
        status = "ready" if key is None and project is not None else statuses.get(key, "not started")
        step_class = "sam-workflow-step"
        if index == 1 and project is None:
            step_class += " is-active"
        elif status == "completed":
            step_class += " is-complete"
        elif status in {"running", "ready"}:
            step_class += " is-active"
        with ui.element("div").classes(step_class):
            ui.label(str(index)).classes("sam-step-number")
            ui.label(label).classes("sam-step-label")
            status_label = ui.label(status.replace("_", " ")).classes("sam-step-status")
            status_labels[key] = status_label
            if key in _PHASE_RANGES and project is not None:
                phase_percent = _phase_percent(key, statuses, progress)
                with ui.row().classes("items-center gap-2 w-full q-mt-xs"):
                    phase_bar = ui.linear_progress(
                        value=phase_percent / 100.0,
                        show_value=False,
                    ).classes("flex-1")
                    phase_label = ui.label(f"{phase_percent:.0f}%").classes("text-xs sam-muted")
                phase_bars[key] = phase_bar
                phase_labels[key] = phase_label

    if project is not None and project.get("workflow", {}).get("classification") == "completed":
        timing = _processing_time_summary(state.active_project_directory)
        if timing is not None:
            with ui.element("div").classes("sam-processing-time"):
                ui.label(f"Rechenzeit gesamt: {_format_elapsed(timing['total_seconds'])}").classes("sam-processing-total")
                for name, seconds in timing["groups"].items():
                    with ui.row().classes("items-center justify-between w-full no-wrap"):
                        ui.label(name).classes("text-xs sam-muted")
                        ui.label(_format_elapsed(seconds)).classes("text-xs sam-muted")
        storage = _storage_summary(state.active_project_directory)
        if storage is not None:
            after = float(storage.get("after_bytes", 0)) / 1024 ** 3
            freed = float(storage.get("freed_bytes", 0)) / 1024 ** 3
            ui.label(f"Speicher: {after:.1f} GB (gespart: {freed:.1f} GB)").classes("text-xs sam-muted")

    if project is not None and _workflow_is_active(project.get("workflow", {})):
        def refresh_workflow() -> None:
            current_project = state.active_project()
            if current_project is None:
                return
            current_statuses = _live_workflow(current_project)
            current_progress = _read_progress(current_project, state.active_project_directory)
            for stage, bar in phase_bars.items():
                value = _phase_percent(stage, current_statuses, current_progress)
                bar.set_value(value / 100.0)
                phase_labels[stage].set_text(f"{value:.0f}%")
            for stage, label in status_labels.items():
                if stage in current_statuses:
                    label.set_text(str(current_statuses[stage]).replace("_", " "))

        ui.timer(1.0, refresh_workflow)


def _phase_percent(stage: str, statuses: dict, progress: dict) -> float:
    """Convert global pipeline progress into an independent phase percentage."""
    if statuses.get(stage) == "completed":
        return 100.0
    members = _PHASE_STAGES.get(stage, ())
    estimator_stages = [item for item in progress.get("stages", []) if item.get("key") in members]
    if estimator_stages:
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
    projects = state.projects.discover_projects()
    with ui.element("section").classes("sam-surface sam-recent"):
        with ui.row().classes("sam-recent-head items-center justify-between w-full"):
            ui.label("Recent projects").classes("sam-section-title")
            ui.button("View all projects", icon="arrow_forward", on_click=lambda: ui.navigate.to("/projects")).props("flat no-caps").classes("text-cyan-4")
        if not projects:
            ui.label("No saved projects yet. Create your first project to begin the workflow.").classes("sam-empty-row")
            return
        for project_directory in projects[:5]:
            project = state.projects.load(project_directory)
            with ui.row().classes("sam-project-row w-full items-center no-wrap"):
                ui.icon("folder").classes("text-cyan-4 q-mr-md")
                with ui.column().classes("gap-0 flex-1").style("min-width: 0"):
                    ui.label(project["project"]["name"]).classes("sam-project-name")
                    ui.label(str(project_directory)).classes("sam-project-path")
                ui.button("Open", on_click=lambda path=project_directory: _open_project(path)).props("flat no-caps").classes("text-cyan-4")


def _open_project(project_directory: Path) -> None:
    state.open_project(project_directory)
    ui.navigate.to("/")


def _start_processing() -> None:
    if state.active_project_directory is None:
        return
    try:
        state.processing.start_pipeline(state.active_project_directory)
        ui.notify("Processing started. Keep SAMcloud open while COLMAP and SAM3 run.")
        ui.navigate.to("/")
    except (OSError, ValueError) as error:
        ui.notify(str(error), type="negative")


def _continue_after_control_points() -> None:
    if state.active_project_directory is None:
        return
    try:
        state.processing.continue_after_control_points(state.active_project_directory)
        ui.notify("Control-point georeferencing and processing started.")
        ui.navigate.to("/")
    except (OSError, ValueError) as error:
        ui.notify(str(error), type="negative")


def _export_las() -> None:
    if state.active_project_directory is None:
        return
    try:
        state.processing.export_las(state.active_project_directory)
        ui.notify("LAS export started. The result will be saved in products/classified.")
        ui.navigate.to("/")
    except (OSError, ValueError) as error:
        ui.notify(str(error), type="negative")
