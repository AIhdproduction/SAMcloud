"""DXF export page: header, preparation, side view (indoor) and DXF review.

Nothing heavy runs automatically: every preview, alignment, level detection, generation and export
is started by an explicit button through DxfPageController.start_job. Settings only mark caches stale.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Callable

from nicegui import run, ui

from samcloud.app.dxf_controller import LAYER_MODES, MODEL_PROFILES, DxfPageController, dxf_unavailable_reason
from samcloud.app.dxf_review_canvas import DxfReviewCanvas, background_url, level_key, ortho_preview_url, payload_url
from samcloud.app.dxf_side_canvas import DxfSideCanvas
from samcloud.app.dxf_side_view import encode_png_data_url, render_side_view
from samcloud.app.i18n import t, translate_or
from samcloud.app.ifc_viewer import IfcViewer, model_url
from samcloud.app.layout import render_page
from samcloud.app.state import state
from samcloud.bim import ui_data as bim_ui_data
from samcloud.dxf.job_runner import DxfJobRunner

ACTION_LABELS: dict[str, str] = {
    "preview": "dxf.action.preview",
    "align": "dxf.action.align",
    "levels": "dxf.action.levels",
    "generate": "dxf.action.generate",
    "export": "dxf.action.export",
    "ortho": "dxf.action.ortho",
    "segment": "dxf.action.segment",
    "clean_plan": "dxf.action.clean_plan",
    "prepare_levels": "dxf.action.prepare_levels",
    "run_all": "dxf.action.run_all",
    "export_ifc": "dxf.action.export_ifc",
    "refine_openings": "dxf.action.refine_openings",
}
"""Job action -> translation key of its display label (resolved at render time by _action_label)."""


def _action_label(action: str, fallback: str | None = None) -> str:
    """Translated display label of a DXF job action; unknown actions show ``fallback`` (default: the action itself)."""
    key = ACTION_LABELS.get(action)
    if key is None:
        return action if fallback is None else fallback
    return t(key)


_STEP_NAME_KEYS: dict[str, str] = {
    "Build orthophoto": "dxf.stage.step_names.ortho",
    "Segment orthophoto": "dxf.stage.step_names.segment",
    "Generate clean plan": "dxf.stage.step_names.clean_plan",
    "Refine openings": "dxf.stage.step_names.refine_openings",
    "Export DXF": "dxf.stage.step_names.export",
    "Export IFC": "dxf.stage.step_names.export_ifc",
}
_STEP_STATE_KEYS: dict[str, str] = {
    "starting": "dxf.stage.states.starting",
    "complete": "dxf.stage.states.complete",
    "skipped (up to date)": "dxf.stage.states.skipped",
}
_STAGE_PATTERN = re.compile(r"^Step (\d+)/(\d+): (.+?) – (.*)$")


_INNER_STAGE_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = tuple(
    (re.compile(pattern), key)
    for pattern, key in (
        (r"^Selecting photographs$", "dxf.stage.inner.selecting_photos"),
        (r"^Detecting windows and doors$", "dxf.stage.inner.detecting_openings"),
        (r"^Detecting openings from points$", "dxf.stage.inner.detecting_openings_points"),
        (r"^Refined openings are up to date$", "dxf.stage.inner.refine_up_to_date"),
        (r"^Refined (?P<count>\d+) openings \((?P<mode>[^)]*)\)$", "dxf.stage.inner.refined"),
        (r"^Reading points (?P<amount>[\d.]+) M$", "dxf.stage.inner.reading_points"),
        (r"^Scoring photographs (?P<index>\d+)/(?P<total>\d+)$", "dxf.stage.inner.scoring_photos"),
        (r"^Voting photographs (?P<index>\d+)/(?P<total>\d+)$", "dxf.stage.inner.voting_photos"),
        (r"^Plan already contains the refined openings$", "dxf.stage.inner.plan_has_openings"),
    )
)


def _inner_stage_label(stage: str) -> str:
    """Translate the inner stage texts of the opening refinement; unknown texts stay as they are."""
    for pattern, key in _INNER_STAGE_PATTERNS:
        match = pattern.match(stage)
        if match is not None:
            return t(key, **match.groupdict())
    return stage


def _stage_label(stage: str) -> str:
    """Translate the combined workflow stage ("Step 4/7: Refine openings - starting"); other stages stay as they are."""
    match = _STAGE_PATTERN.match(stage)
    if match is None:
        return _inner_stage_label(stage)
    index, total, name, state = match.groups()
    name_key = _STEP_NAME_KEYS.get(name)
    state_key = _STEP_STATE_KEYS.get(state)
    return t(
        "dxf.stage.step",
        index=index, total=total,
        name=t(name_key) if name_key else name,
        stage=t(state_key) if state_key else _inner_stage_label(state),
    )


def _job_state_label(job_state: str) -> str:
    """Translated job state (done, failed, cancelled, ...); unknown states are shown as-is."""
    return translate_or(f"dxf.job_state.{job_state}", job_state)


def _cache_status_label(status: str) -> str:
    """Translated cache freshness (missing / ok / stale)."""
    return translate_or(f"dxf.cache_status.{status}", status)


def _ifc_cache_status(controller: DxfPageController, level_id: str | None) -> str:
    """Freshness of the IFC export of a level in the cache-status vocabulary (missing / ok / stale)."""
    status = bim_ui_data.ifc_status(controller.project_dir, level_key(level_id))
    return "ok" if status == "fresh" else status


def _refine_cache_status(controller: DxfPageController, level_id: str | None) -> str:
    """Freshness of the opening refinement of a level in the cache-status vocabulary (missing / ok / stale)."""
    from samcloud.dxf.jobs import refine_openings_status

    try:
        status = refine_openings_status(controller.project_dir, level_id)
    except Exception:  # noqa: BLE001 - an unreadable cache counts as stale in the checklist
        return "stale"
    return "ok" if status == "fresh" else status


def _mode_label(mode: str) -> str:
    """Translated project mode value (indoor / outdoor)."""
    return translate_or(f"dxf.mode_value.{mode}", mode)

_RUNNERS: dict[Path, DxfJobRunner] = {}

_STYLE = """
<style>
.sam-dxf-banner { padding:10px 14px; border-radius:7px; font-size:14px; }
.sam-dxf-banner.is-warning { background:rgba(255,183,77,.13); border:1px solid #a8752b; color:#ffd59a; }
.sam-dxf-banner.is-error { background:rgba(239,107,115,.13); border:1px solid #9b3c43; color:#ffc2c6; }
.sam-dxf-banner.is-info { background:rgba(38,183,239,.10); border:1px solid #24719b; color:#bfe9fb; }
.sam-dxf-swatch { width:14px; height:14px; border-radius:3px; border:1px solid rgba(255,255,255,.35); flex:none; }
.sam-result-fullscreen { position:fixed!important; inset:0; z-index:6000; background:var(--sam-bg); padding:12px; width:100vw!important; height:100vh!important; }
.sam-result-fullscreen .sam-result-canvas-box { height:100%!important; }
.sam-result-fullscreen .sam-result-side { overflow-y:auto; overflow-x:hidden; min-width:0; max-height:100%; flex:none; }
.sam-result-side .q-checkbox__label { overflow:hidden; text-overflow:ellipsis; white-space:nowrap; min-width:0; }
.q-slider.q-slider--disabled, .q-slider.disabled { opacity:.38!important; cursor:not-allowed!important; filter:grayscale(1); }
.sam-dxf-log { max-height:220px; overflow:auto; white-space:pre-wrap; font-family:monospace; font-size:11px; color:#c9d8e4; background:#050b12; border:1px solid #20394b; border-radius:6px; padding:8px; }
</style>
"""


def runner_for(project_directory: Path) -> DxfJobRunner:
    """One job runner per project, shared across page visits so a running Popen stays tracked."""
    key = Path(project_directory).resolve()
    if key not in _RUNNERS:
        _RUNNERS[key] = DxfJobRunner(key)
    return _RUNNERS[key]


def render() -> None:
    """Render the /dxf page for the active project."""

    def content() -> None:
        project_directory = state.active_project_directory
        if project_directory is None or state.active_project() is None:
            ui.label(t("dxf.page.no_project")).classes("sam-muted")
            return
        if dxf_unavailable_reason(project_directory) is not None:
            ui.label(t("dxf.page.not_ready")).classes("sam-muted").mark("dxf-not-ready")
            return
        controller = DxfPageController(project_directory, runner=runner_for(project_directory))
        DxfPage(controller).build()

    render_page(t("dxf.page.title"), content, current_path="/dxf", wide=True)


def _banner(text: str, kind: str = "warning") -> ui.label:
    return ui.label(text).classes(f"sam-dxf-banner is-{kind} w-full")


class DxfPage:
    """Widget glue around DxfPageController; all decisions live in the controller."""

    def __init__(self, controller: DxfPageController) -> None:
        self.controller = controller
        self.job_buttons: list[tuple[ui.button, bool]] = []
        self.was_running = controller.is_running()
        targets = controller.review_targets()
        self.review_level: str | None = targets[0] if targets else None
        levels = controller.available_levels()
        self.compute_level: str | None = str(levels[0]["id"]) if controller.mode == "indoor" and len(levels) == 1 else None
        self.side_canvas: DxfSideCanvas | None = None
        self.side_scale: float = controller.scale_factor
        self.side_rendered = False
        self.review_canvas: DxfReviewCanvas | None = None
        self.result_canvas: DxfReviewCanvas | None = None
        self.result_fullscreen = False
        self.result_row: ui.row | None = None
        self.result_opacity_slider: ui.slider | None = None
        self.run_all_button: ui.button | None = None
        self.hidden_layers: set[str] = set()
        self.measurement: dict[str, Any] | None = None
        self.job_stage: ui.label | None = None
        self.job_progress: ui.linear_progress | None = None
        self.cancel_button: ui.button | None = None
        self.job_log: ui.label | None = None
        self.measure_label: ui.label | None = None

    # ------------------------------------------------------------------ frame

    def build(self) -> None:
        ui.add_head_html(_STYLE)
        with ui.column().classes("w-full gap-4"):
            self.body()
        ui.timer(1.0, self.poll)

    @ui.refreshable_method
    def body(self) -> None:
        self.job_buttons = []
        self.side_canvas = None
        self.review_canvas = None
        self.result_canvas = None
        self.result_row = None
        self.result_opacity_slider = None
        self.result_fullscreen = False
        self.ifc_viewer = None
        self.ifc_row = None
        self.ifc_fullscreen = False
        self.run_all_button = None
        self.job_stage = self.job_progress = self.cancel_button = self.job_log = None
        self.build_header()
        if self.controller.profile_required:
            self.build_profile_required()
            return
        blocked = self.controller.blocked_message
        if blocked is not None:
            _banner(blocked, "error").props("data-testid=blocked")
            return
        self.build_compute_card()
        self.build_result_card()
        self.build_ifc_card()
        with ui.expansion(t("dxf.page.advanced"), icon="tune", value=False).classes("w-full").mark("advanced"):
            self.stale_banner()
            self.build_preparation()
            if self.controller.mode == "indoor":
                self.build_side_view()
            self.build_review_section()
        self.apply_busy(self.controller.is_running())

    def job_button(self, label: str, icon: str, on_click: Callable[[], Any], enabled: bool = True) -> ui.button:
        button = ui.button(label, icon=icon, on_click=on_click).props("no-caps unelevated").classes("sam-button-secondary")
        self.job_buttons.append((button, enabled))
        return button

    def apply_busy(self, running: bool) -> None:
        self.job_buttons = [(button, enabled) for button, enabled in self.job_buttons if not button.is_deleted]
        for button, enabled in self.job_buttons:
            button.set_enabled(enabled and not running)
        if self.cancel_button is not None:
            self.cancel_button.set_visibility(running)

    # ------------------------------------------------------------------ header

    def build_header(self) -> None:
        controller = self.controller
        with ui.card().classes("w-full"):
            with ui.row().classes("items-center w-full gap-6"):
                ui.label(controller.project_name).classes("sam-section-title")
                ui.label(t("dxf.header.mode", mode=_mode_label(controller.mode))).classes("sam-muted").props("data-testid=mode")
                ui.label(t("dxf.header.profile", profile=controller.profile or t("dxf.header.not_set"))).classes("sam-muted")
                ui.label(t("dxf.header.scale", status=controller.scale_status)).classes("sam-muted")
            ui.label(t("dxf.header.point_source", source=controller.source_label)).classes("sam-muted text-sm").mark("point-source")
            with ui.row().classes("items-end w-full gap-3"):
                current = controller.las_path
                # The classified PLY is used automatically; the field is only for choosing another LAS file.
                shown = "" if current is None or controller.source_is_ply else str(current)
                las_input = ui.input(t("dxf.header.las_file"), value=shown, placeholder=t("dxf.header.las_placeholder")).classes("flex-1").props("outlined dense").mark("las-path")
                ui.button(t("dxf.header.use_las"), icon="check", on_click=lambda: self.choose_las(las_input.value)).props("flat no-caps")
                if controller.las_is_custom:
                    ui.button(t("dxf.header.project_cloud"), icon="restart_alt", on_click=lambda: self.choose_las(None)).props("flat no-caps")
                mode_options = {"auto": t("dxf.header.mode_auto"), "indoor": t("dxf.header.mode_indoor"), "outdoor": t("dxf.header.mode_outdoor")}
                ui.select(mode_options, label=t("dxf.header.mode_label"), value=controller.state.mode_override or "auto",
                          on_change=lambda event: self.set_mode(event.value)).classes("w-40").props("outlined dense").mark("mode-select")
                if controller.state.profile_name and not controller.profile_required:
                    ui.select(list(MODEL_PROFILES), label=t("dxf.header.profile_label"), value=controller.state.profile_name,
                              on_change=lambda event: self.set_profile(event.value)).classes("w-48").props("outlined dense").mark("profile-change")
            if current is None or not current.is_file():
                _banner(t("dxf.header.no_cloud"), "warning")

    def build_profile_required(self) -> None:
        with ui.card().classes("w-full"):
            _banner(t("dxf.header.profile_missing"), "info")
            ui.select(list(MODEL_PROFILES), label=t("dxf.header.profile_required"), value=None,
                      on_change=lambda event: self.set_profile(event.value)).classes("w-72").props("outlined dense").mark("profile-required")

    def choose_las(self, path: str | None) -> None:
        try:
            self.controller.set_las_path(path.strip() if path else None)
        except ValueError as error:
            ui.notify(str(error), type="negative")
            return
        self.body.refresh()

    def set_mode(self, value: str | None) -> None:
        self.controller.set_mode_override(None if value in (None, "auto") else value)
        targets = self.controller.review_targets()
        self.review_level = targets[0] if targets else None
        self.body.refresh()

    def set_profile(self, value: str | None) -> None:
        if not value:
            return
        self.controller.set_profile(value)
        self.body.refresh()

    # ------------------------------------------------------------------ jobs

    def build_job_bar(self) -> None:
        status = self.controller.job_status()
        with ui.row().classes("items-center w-full gap-3 no-wrap"):
            self.job_stage = ui.label("").classes("text-sm").mark("job-stage")
            self.job_progress = ui.linear_progress(value=0.0, show_value=False).classes("flex-1")
            self.cancel_button = ui.button(t("common.cancel"), icon="stop", on_click=self.cancel).props("flat no-caps color=negative")
        self.job_log = ui.label("").classes("sam-dxf-log w-full")
        self.update_job_bar(status, self.controller.is_running())

    def update_job_bar(self, status: dict[str, Any], running: bool) -> None:
        if self.job_stage is None or self.job_progress is None or self.job_log is None:
            return
        raw_action = str(status.get("action", ""))
        action = _action_label(raw_action, raw_action or t("dxf.job.fallback"))
        job_state = str(status.get("state", "idle"))
        if running:
            self.job_stage.set_text(t("dxf.job.running", action=action, stage=_stage_label(str(status.get("stage", ""))),
                                      percent=f"{float(status.get('fraction', 0.0)) * 100:.0f}"))
        elif job_state == "idle":
            self.job_stage.set_text(t("dxf.job.idle"))
        else:
            message = f" - {status['message']}" if status.get("message") else ""
            self.job_stage.set_text(t("dxf.job.last", action=action, state=_job_state_label(job_state), message=message))
        self.job_progress.set_value(float(status.get("fraction", 0.0)) if running or job_state == "done" else 0.0)
        show_log = not running and job_state in ("failed", "cancelled", "unknown")
        self.job_log.set_text("\n".join(self.controller.log_tail(25)) if show_log else "")
        self.job_log.set_visibility(show_log)
        if self.cancel_button is not None:
            self.cancel_button.set_visibility(running)

    def start(self, action: str, level_id: str | None = None, *, force: bool = False) -> None:
        try:
            self.controller.start_job(action, level_id, force=force)
        except (ValueError, RuntimeError) as error:
            ui.notify(str(error), type="negative")
            return
        self.was_running = True
        self.apply_busy(True)
        self.update_job_bar(self.controller.job_status(), True)
        ui.notify(t("dxf.job.started", action=_action_label(action)))

    async def cancel(self) -> None:
        if self.cancel_button is not None:
            self.cancel_button.disable()
        await run.io_bound(self.controller.cancel_job)
        self.poll()

    def poll(self) -> None:
        running = self.controller.is_running()
        status = self.controller.job_status()
        if self.was_running and not running:
            self.was_running = False
            self.controller.refresh_state()
            if str(status.get("action", "")) == "run_all" and str(status.get("state", "")) == "done":
                self.controller.set_plan_variant("ortho")
            levels = self.controller.available_levels()
            if self.controller.mode == "indoor" and self.compute_level is None and len(levels) == 1:
                self.compute_level = str(levels[0]["id"])
            job_state = str(status.get("state", ""))
            action = _action_label(str(status.get("action", "")), t("dxf.job.fallback"))
            kind = "positive" if job_state == "done" else ("warning" if job_state == "cancelled" else "negative")
            ui.notify(t("dxf.job.finished", action=action, state=_job_state_label(job_state or "finished")), type=kind)
            targets = self.controller.review_targets()
            if self.review_level not in targets:
                self.review_level = targets[0] if targets else None
            self.body.refresh()
            return
        self.was_running = running
        self.update_job_bar(status, running)
        self.apply_busy(running)

    # ------------------------------------------------------------------ simple workflow

    def primary_job_button(self, label: str, icon: str, on_click: Callable[[], Any], enabled: bool = True) -> ui.button:
        button = self.job_button(label, icon, on_click, enabled=enabled)
        button.props("color=primary").classes("text-lg")
        return button

    def build_compute_card(self) -> None:
        controller = self.controller
        with ui.card().classes("w-full gap-3"):
            ui.label(t("dxf.compute.title")).classes("sam-section-title")
            force = ui.checkbox(t("dxf.compute.force"), value=False).classes("text-sm")
            if controller.mode == "outdoor":
                self.run_all_button = self.primary_job_button(
                    t("dxf.compute.run_all"), "play_arrow", lambda: self.start("run_all", force=bool(force.value)),
                )
                self.run_all_button.mark("run-all")
                self.build_step_checklist(None)
            else:
                levels = controller.available_levels()
                if not levels:
                    self.primary_job_button(
                        t("dxf.compute.detect_storeys"), "layers", lambda: self.start("prepare_levels", force=bool(force.value)),
                    ).mark("prepare-levels")
                    self.build_step_checklist(None)
                else:
                    options = {
                        str(level["id"]): f"{level['name']} ({float(level['floor_z']):.2f} m)"
                        for level in levels
                    }
                    ui.toggle(options, value=self.compute_level if self.compute_level in options else None,
                              on_change=lambda event: self.select_compute_level(event.value)).mark("storey-picker")
                    label = t("dxf.compute.run_all") if self.compute_level is None else t("dxf.compute.run_all_for", level=options[self.compute_level])
                    self.run_all_button = self.primary_job_button(
                        label, "play_arrow", lambda: self.start("run_all", self.compute_level, force=bool(force.value)),
                        enabled=self.compute_level is not None,
                    )
                    self.run_all_button.mark("run-all")
                    self.job_button(t("dxf.compute.detect_again"), "refresh", lambda: self.start("prepare_levels", force=bool(force.value))).props("flat")
                    self.build_step_checklist(self.compute_level)
            self.build_job_bar()

    def select_compute_level(self, level_id: str | None) -> None:
        self.compute_level = level_id
        self.review_level = level_id
        self.hidden_layers = set()
        if self.run_all_button is None:
            return
        level = next((item for item in self.controller.available_levels() if item["id"] == level_id), None)
        label = t("dxf.compute.run_all") if level is None else t("dxf.compute.run_all_for", level=f"{level['name']} ({float(level['floor_z']):.2f} m)")
        enabled = level is not None
        self.run_all_button.set_text(label)
        self.run_all_button.set_enabled(enabled and not self.controller.is_running())
        self.job_buttons = [
            (button, enabled if button is self.run_all_button else button_enabled)
            for button, button_enabled in self.job_buttons
        ]

    def build_step_checklist(self, level_id: str | None) -> None:
        controller = self.controller
        if controller.mode == "indoor" and level_id is None:
            ui.label(t("dxf.compute.select_storey")).classes("sam-muted text-sm")
            return
        rows = [
            (t("dxf.compute.step_ortho"), controller.ortho_status(level_id)),
            (t("dxf.compute.step_segmentation"), controller.segment_status(level_id)),
            (t("dxf.compute.step_clean_plan"), controller.plan_status(level_id, "ortho")),
            *([(t("dxf.compute.step_refine"), _refine_cache_status(controller, level_id))] if controller.mode == "indoor" else []),
            (t("dxf.compute.step_export"), controller.plan_status(level_id, "ortho")),
            (t("dxf.compute.step_ifc"), _ifc_cache_status(controller, level_id)),
        ]
        with ui.column().classes("gap-1"):
            for name, current in rows:
                ui.label(t("dxf.compute.step_row", name=name, status=_cache_status_label(current))).classes("sam-muted text-sm")

    # ------------------------------------------------------------------ stale + preparation

    @ui.refreshable_method
    def stale_banner(self) -> None:
        text = self.controller.stale_banner_text()
        if text:
            _banner(text).mark("stale-banner")

    def settings_changed(self) -> None:
        """Refresh only the staleness indicators; never regenerates anything."""
        self.stale_banner.refresh()
        self.review_status.refresh()

    def build_preparation(self) -> None:
        controller = self.controller
        indoor = controller.mode == "indoor"
        with ui.card().classes("w-full"):
            ui.label(t("dxf.prep.title")).classes("sam-section-title")
            with ui.row().classes("items-center gap-3"):
                self.job_button(t("dxf.action.preview"), "grain", lambda: self.start("preview"))
                if indoor:
                    self.job_button(t("dxf.action.align"), "straighten", lambda: self.start("align"))
                    self.job_button(t("dxf.action.levels"), "layers", lambda: self.start("levels"))
            ui.label(t("dxf.prep.preview_available") if controller.has_preview else t("dxf.prep.preview_missing")).classes("sam-muted text-sm")
            for warning in controller.plausibility_warnings:
                _banner(warning, "warning")
            with ui.row().classes("items-end gap-3"):
                scale_input = ui.number(t("dxf.prep.scale_factor"), value=controller.scale_factor, min=0.0, format="%.6g").classes("w-40").props("outlined dense").mark("scale-factor")
                ui.button(t("common.apply"), on_click=lambda: self.apply_scale(scale_input.value)).props("flat no-caps").mark("scale-apply")
                scale_source = translate_or(f"dxf.scale_source.{controller.scale_source}", controller.scale_source)
                ui.label(t("dxf.prep.scale_line", source=scale_source, status=controller.scale_status)).classes("sam-muted text-sm")
            ui.label(t("dxf.prep.measure_hint")).classes("sam-muted text-xs")
            if indoor:
                alignment = controller.alignment
                if alignment is None:
                    ui.label(t("dxf.prep.alignment_none")).classes("sam-muted text-sm")
                else:
                    with ui.row().classes("items-center gap-4"):
                        ui.label(t("dxf.prep.alignment", tilt=f"{float(alignment.get('tilt_degrees', 0.0)):.2f}",
                                   yaw=f"{float(alignment.get('yaw_degrees', 0.0)):.2f}", method=alignment.get("method", ""))).classes("text-sm")
                        ui.switch(t("dxf.prep.align_walls"), value=controller.align_walls,
                                  on_change=lambda event: self.set_align_walls(event.value)).mark("align-walls")

    def apply_scale(self, value: float | None) -> None:
        try:
            self.controller.set_scale_factor(float(value) if value is not None else float("nan"))
        except ValueError as error:
            ui.notify(str(error), type="negative")
            return
        ui.notify(t("dxf.prep.scale_applied"))
        self.body.refresh()

    def set_align_walls(self, enabled: bool) -> None:
        try:
            self.controller.set_align_walls(bool(enabled))
        except ValueError as error:
            ui.notify(str(error), type="negative")
            return
        self.settings_changed()

    # ------------------------------------------------------------------ side view

    def build_side_view(self) -> None:
        controller = self.controller
        view = controller.state.view
        with ui.card().classes("w-full"):
            ui.label(t("dxf.side.title")).classes("sam-section-title")
            with ui.row().classes("items-end gap-3 w-full"):
                ui.toggle({"x": t("dxf.side.view_x"), "y": t("dxf.side.view_y")}, value=view.get("side_axis", "x"),
                          on_change=lambda event: self.set_side_option("side_axis", event.value)).mark("side-axis")
                ui.number(t("dxf.side.slab_center"), value=view.get("slab_center"),
                          on_change=lambda event: self.set_side_option("slab_center", event.value)).classes("w-36").props("outlined dense clearable").mark("slab-center")
                ui.number(t("dxf.side.slab_thickness"), value=view.get("slab_thickness"), min=0.0,
                          on_change=lambda event: self.set_side_option("slab_thickness", event.value)).classes("w-36").props("outlined dense clearable").mark("slab-thickness")
                ui.button(t("dxf.side.render"), icon="visibility", on_click=self.render_side).props("flat no-caps").set_enabled(controller.has_preview)
                ui.toggle({"pan": t("dxf.side.tool_pan"), "add": t("dxf.side.tool_add"), "measure": t("dxf.view.measure")}, value="pan",
                          on_change=lambda event: self.side_canvas.set_tool(event.value) if self.side_canvas else None).mark("side-tool")
                ui.button(t("dxf.view.fit"), on_click=lambda: self.side_canvas.fit() if self.side_canvas else None).props("flat no-caps")
            with ui.row().classes("w-full gap-4 no-wrap items-stretch"):
                with ui.element("div").classes("flex-1").style("height:520px"):
                    self.side_canvas = DxfSideCanvas().classes("w-full h-full").mark("side-canvas")
                    self.side_canvas.set_levels(controller.available_levels())
                    self.side_canvas.on("level_move", self.on_level_move)
                    self.side_canvas.on("add_level", self.on_add_level)
                    self.side_canvas.on("measure", lambda event: self.open_measure_dialog(event.args, None))
                with ui.column().classes("w-96 gap-2"):
                    self.levels_panel()
        self.side_rendered = False

    @ui.refreshable_method
    def levels_panel(self) -> None:
        levels = self.controller.available_levels()
        if not levels:
            ui.label(t("dxf.side.no_levels")).classes("sam-muted text-sm")
            return
        ui.label(t("dxf.side.levels_hint")).classes("sam-muted text-xs")
        for level in sorted(levels, key=lambda item: float(item["floor_z"])):
            level_id = str(level["id"])
            with ui.row().classes("items-center gap-1 no-wrap w-full"):
                ui.checkbox(value=bool(level.get("checked", True)),
                            on_change=lambda event, lid=level_id: self.level_edit(self.controller.set_level_checked, lid, bool(event.value))).mark(f"level-check-{level_id}")
                name_input = ui.input(value=str(level["name"])).classes("w-16").props("dense outlined").mark(f"level-name-{level_id}")
                name_input.on("blur", lambda _event, lid=level_id, field=name_input: self.rename(lid, field.value))
                name_input.on("keydown.enter", lambda _event, lid=level_id, field=name_input: self.rename(lid, field.value))
                ui.number(t("dxf.side.floor"), value=round(float(level["floor_z"]), 3), format="%.3f",
                          on_change=lambda event, lid=level_id: self.level_edit(self.controller.move_level, lid, floor_z=event.value) if event.value is not None else None).classes("w-24").props("dense outlined").mark(f"level-floor-{level_id}")
                ui.number(t("dxf.side.ceiling"), value=None if level.get("ceiling_z") is None else round(float(level["ceiling_z"]), 3), format="%.3f",
                          on_change=lambda event, lid=level_id: self.level_edit(self.controller.move_level, lid, ceiling_z=event.value) if event.value is not None else None).classes("w-24").props("dense outlined").mark(f"level-ceiling-{level_id}")
                ui.button(icon="delete", on_click=lambda lid=level_id: self.level_edit(self.controller.delete_level, lid)).props("flat dense round color=negative").mark(f"level-delete-{level_id}")

    def level_edit(self, action: Callable[..., Any], *args: Any, **kwargs: Any) -> None:
        try:
            action(*args, **kwargs)
        except ValueError as error:
            ui.notify(str(error), type="negative")
            return
        self.after_level_change()

    def rename(self, level_id: str, name: str | None) -> None:
        cleaned = (name or "").strip()
        current = next((lvl for lvl in self.controller.available_levels() if lvl["id"] == level_id), None)
        if current is None or not cleaned or cleaned == current["name"]:
            return
        self.level_edit(self.controller.rename_level, level_id, cleaned)

    def after_level_change(self) -> None:
        if self.side_canvas is not None:
            self.side_canvas.set_levels(self.controller.available_levels())
        self.levels_panel.refresh()
        targets = self.controller.review_targets()
        if self.review_level not in targets:
            self.review_level = targets[0] if targets else None
            self.review_section.refresh()
        else:
            self.review_level_select.refresh()
        self.settings_changed()

    def on_level_move(self, event: Any) -> None:
        args = event.args or {}
        edge = "floor_z" if args.get("edge") == "floor" else "ceiling_z"
        self.level_edit(self.controller.move_level, str(args.get("id")), **{edge: float(args.get("z"))})

    def on_add_level(self, event: Any) -> None:
        z = float((event.args or {}).get("z"))
        self.level_edit(self.controller.add_level_at, z)

    def set_side_option(self, key: str, value: Any) -> None:
        self.controller.set_view_option(key, value)
        if self.side_rendered:
            self.render_side_later()

    def render_side_later(self) -> None:
        ui.timer(0.0, self.render_side, once=True)

    async def render_side(self) -> None:
        controller = self.controller
        preview = controller.project_dir / "products" / "dxf" / "preview.npz"
        if not preview.is_file():
            ui.notify(t("dxf.side.preview_first"), type="warning")
            return
        view = controller.state.view
        axis = str(view.get("side_axis", "x"))
        scale = controller.scale_factor
        result = await run.io_bound(render_side_view, preview, axis, view.get("slab_center"), view.get("slab_thickness"),
                                    controller.state.alignment, scale)
        if result is None or self.side_canvas is None:
            return
        png, bounds = result
        self.side_scale = scale
        self.side_rendered = True
        self.side_canvas.set_image(encode_png_data_url(png), bounds, "Y" if axis == "x" else "X")
        self.side_canvas.set_levels(controller.available_levels())

    # ------------------------------------------------------------------ measure → scale

    def open_measure_dialog(self, args: dict[str, Any], points_scale: float | None) -> None:
        measured = float(args.get("distance", 0.0))
        self.measurement = {"p1": args.get("p1"), "p2": args.get("p2"), "distance": measured}
        with ui.dialog() as dialog, ui.card():
            ui.label(t("dxf.measure.measured_distance", distance=f"{measured:.4f}")).classes("text-md")
            real = ui.number(t("dxf.measure.real_distance"), value=None, min=0.0).props("outlined dense autofocus").mark("real-distance")

            def apply() -> None:
                try:
                    factor = self.controller.measure_scale(args["p1"], args["p2"], float(real.value or 0.0),
                                                           points_scale=points_scale if points_scale is not None else self.side_scale)
                except (ValueError, KeyError, TypeError) as error:
                    ui.notify(str(error), type="negative")
                    return
                dialog.close()
                ui.notify(t("dxf.measure.scale_applied", factor=f"{factor:.6g}"))
                self.body.refresh()

            with ui.row():
                ui.button(t("dxf.measure.use_for_scale"), on_click=apply).props("unelevated no-caps").mark("apply-measure")
                ui.button(t("common.close"), on_click=dialog.close).props("flat no-caps")
        dialog.open()

    # ------------------------------------------------------------------ review

    def build_result_card(self) -> None:
        controller = self.controller
        level = self.compute_level if controller.mode == "indoor" else None
        result = controller.plan_result(level, "ortho") if (controller.mode != "indoor" or level is not None) else None
        with ui.card().classes("w-full gap-3"):
            ui.label(t("dxf.result.title")).classes("sam-section-title")
            if result is None:
                ui.label(t("dxf.result.placeholder")).classes("sam-muted").mark("result-placeholder")
                return
            self.build_result_canvas(result, level)
            with ui.row().classes("items-center gap-2"):
                ui.button(t("dxf.result.open_dxf_folder"), icon="folder_open", on_click=self.open_dxf_folder).props("flat no-caps")
                ui.button(t("dxf.result.download_dxf"), icon="download", on_click=lambda: self.download_dxf(level)).props("flat no-caps")
                has_ifc = bim_ui_data.find_ifc(controller.project_dir, level_key(level)) is not None
                ifc_button = ui.button(t("ifc.card.download"), icon="download", on_click=lambda: self.download_ifc(level)).props("flat no-caps").mark("download-ifc")
                ifc_button.set_enabled(has_ifc)
                if not has_ifc:
                    with ifc_button:
                        ui.tooltip(t("ifc.card.download_missing"))
                ui.button(t("dxf.view.save_snapshot"), icon="photo_camera", on_click=lambda: self.save_snapshot("ortho", level)).props("flat no-caps")

    def build_result_canvas(self, result: Any, level: str | None) -> None:
        controller = self.controller
        view = controller.state.view
        background = str(view.get("result_background", "ortho"))
        ortho = controller.ortho_result(level)
        if background == "ortho":
            image = ortho_preview_url(controller, level)
        elif background == "classes":
            image = background_url(controller, level, "classes")
        else:
            image = None
        bounds = ortho.grid.bounds if ortho is not None else result.background.bounds
        self.result_fullscreen = False
        with ui.row().classes("w-full gap-4 no-wrap items-stretch") as result_row:
            self.result_row = result_row
            with ui.element("div").classes("flex-1 relative sam-result-canvas-box").style("height:640px;min-width:0"):
                self.result_canvas = DxfReviewCanvas().classes("w-full h-full").mark("result-canvas")
                self.result_canvas.show(payload_url(controller, level, "ortho"), image, bounds)
                self.result_canvas.set_opacity(self._result_image_opacity(background), 1.0)
                self.result_canvas.set_hidden_layers(sorted(self.hidden_layers))
                self.result_canvas.on("measure", lambda event: self.on_review_measure(event.args, result.scale_factor))
                self.result_fullscreen_button = ui.button(icon="fullscreen", on_click=self.toggle_result_fullscreen).props(
                    "flat round dense color=white").classes("absolute").style(
                    "top:8px;left:8px;z-index:10;background:rgba(9,19,29,.65)").mark("result-fullscreen")
                with self.result_fullscreen_button:
                    self.result_fullscreen_tooltip = ui.tooltip(t("dxf.result.fullscreen"))
                ui.keyboard(on_key=self.on_result_key, ignore=["input", "select", "textarea"])
            with ui.column().classes("w-80 gap-2 sam-result-side"):
                with ui.row().classes("items-center gap-2"):
                    ui.button(t("dxf.view.fit"), on_click=lambda: self.result_canvas.fit() if self.result_canvas else None).props("flat no-caps")
                    ui.toggle({"pan": t("dxf.view.pan"), "measure": t("dxf.view.measure")}, value="pan",
                              on_change=lambda event: self.result_canvas.set_tool(event.value) if self.result_canvas else None)
                ui.toggle({"ortho": t("dxf.view.orthophoto"), "none": t("dxf.result.background_none"), "classes": t("dxf.view.class_colours")}, value=background,
                          on_change=lambda event: self.set_result_background(event.value, level)).mark("result-background")
                ui.label(t("dxf.result.ortho_opacity")).classes("text-sm")
                self.result_opacity_slider = ui.slider(
                    min=0, max=100, step=1, value=round(float(view.get("result_image_opacity", 1.0)) * 100),
                    on_change=lambda event: self.set_result_opacity(event.value)).props("label").mark("result-image-opacity")
                self.result_opacity_slider.set_enabled(background != "none")
                ui.label(t("dxf.view.layers")).classes("text-sm q-mt-sm")
                for name, status in result.statuses.items():
                    color = ",".join(str(value) for value in _layer_rgb(result, name))
                    with ui.row().classes("items-center gap-2 no-wrap w-full"):
                        ui.element("div").classes("sam-dxf-swatch").style(f"background:rgb({color})")
                        ui.checkbox(name, value=name not in self.hidden_layers,
                                    on_change=lambda event, layer=name: self.toggle_result_layer(layer, bool(event.value))).mark(f"result-layer-{name}")
                        ui.label(str(status.entity_count)).classes("sam-muted text-xs")

    def _result_image_opacity(self, background: str) -> float:
        if background == "none":
            return 1.0
        return max(0.0, min(1.0, float(self.controller.state.view.get("result_image_opacity", 1.0))))

    def set_result_background(self, source: str, level: str | None) -> None:
        self.controller.set_view_option("result_background", source)
        slider = self.result_opacity_slider
        if slider is not None:
            slider.set_enabled(source != "none")
        if self.result_canvas is None:
            return
        if source == "ortho":
            image = ortho_preview_url(self.controller, level)
        elif source == "classes":
            image = background_url(self.controller, level, "classes")
        else:
            image = None
        self.result_canvas.set_image(image)
        self.result_canvas.set_opacity(self._result_image_opacity(source), 1.0)

    def set_result_opacity(self, percent: float | None) -> None:
        """Orthophoto/background opacity of the result view; the DXF is always drawn fully opaque."""
        value = max(0.0, min(1.0, float(percent or 0.0) / 100.0))
        self.controller.set_view_option("result_image_opacity", value)
        if self.result_canvas is not None:
            background = str(self.controller.state.view.get("result_background", "ortho"))
            self.result_canvas.set_opacity(self._result_image_opacity(background), 1.0)

    def toggle_result_fullscreen(self) -> None:
        self.set_result_fullscreen(not self.result_fullscreen)

    def set_result_fullscreen(self, active: bool) -> None:
        """Expand the result row (canvas + controls) over the viewport with a CSS overlay."""
        row = self.result_row
        if row is None:
            return
        self.result_fullscreen = active
        if active:
            row.classes(add="sam-result-fullscreen")
        else:
            row.classes(remove="sam-result-fullscreen")
        self.result_fullscreen_button.props(f"icon={'fullscreen_exit' if active else 'fullscreen'}")
        self.result_fullscreen_tooltip.set_text(t("dxf.result.exit_fullscreen") if active else t("dxf.result.fullscreen"))

    def on_result_key(self, event: Any) -> None:
        if self.result_fullscreen and event.action.keydown and event.key.escape:
            self.set_result_fullscreen(False)

    def toggle_result_layer(self, layer: str, visible: bool) -> None:
        if visible:
            self.hidden_layers.discard(layer)
        else:
            self.hidden_layers.add(layer)
        if self.result_canvas is not None:
            self.result_canvas.set_hidden_layers(sorted(self.hidden_layers))

    def open_dxf_folder(self) -> None:
        folder = self.controller.project_dir / "products" / "dxf" / "export"
        if not folder.is_dir():
            ui.notify(t("dxf.result.no_export_folder"), type="warning")
            return
        try:
            os.startfile(str(folder))
        except OSError as error:
            ui.notify(t("dxf.result.open_dxf_folder_failed", error=error), type="negative")

    def download_dxf(self, level: str | None) -> None:
        target_name = self.controller.level_label(level)
        export_dir = self.controller.project_dir / "products" / "dxf" / "export"
        matches = sorted(export_dir.glob(f"*{target_name}*clean.dxf")) if export_dir.is_dir() else []
        if not matches:
            ui.notify(t("dxf.result.no_clean_dxf"), type="warning")
            return
        target = matches[-1]
        ui.download(target.read_bytes(), filename=target.name, media_type="application/dxf")

    def download_ifc(self, level: str | None) -> None:
        target = bim_ui_data.find_ifc(self.controller.project_dir, level_key(level))
        if target is None:
            ui.notify(t("ifc.card.no_file"), type="warning")
            return
        ui.download(target.read_bytes(), filename=target.name, media_type="application/x-step")

    # ------------------------------------------------------------------ IFC viewer

    def build_ifc_card(self) -> None:
        controller = self.controller
        level = self.compute_level if controller.mode == "indoor" else None
        with ui.card().classes("w-full gap-3").mark("ifc-card"):
            ui.label(t("ifc.card.title")).classes("sam-section-title")
            if controller.mode == "indoor" and level is None:
                ui.label(t("ifc.card.empty")).classes("sam-muted").mark("ifc-empty")
                return
            key = level_key(level)
            glb = bim_ui_data.glb_path(controller.project_dir, key)
            data = bim_ui_data.props(controller.project_dir, key)
            if glb is None or not data:
                ui.label(t("ifc.card.empty")).classes("sam-muted").mark("ifc-empty")
                return
            by_node, counts = bim_ui_data.category_index(data)
            self.ifc_hidden = {name for name in bim_ui_data.HIDDEN_BY_DEFAULT if name in counts}
            self.ifc_checks: dict[str, ui.checkbox] = {}
            if bim_ui_data.ifc_status(controller.project_dir, key) == "stale":
                _banner(t("ifc.card.stale"), "warning").mark("ifc-stale")
            qa = bim_ui_data.qa(controller.project_dir, key)
            if isinstance(qa.get("ids_failures"), int):
                ui.label(t("ifc.card.qa_failures", count=qa["ids_failures"])).classes("sam-muted text-sm").mark("ifc-qa")
            self.ifc_fullscreen = False
            with ui.row().classes("w-full gap-4 no-wrap items-stretch") as ifc_row:
                self.ifc_row = ifc_row
                with ui.element("div").classes("flex-1 relative sam-result-canvas-box").style("height:520px;min-width:0"):
                    self.ifc_viewer = IfcViewer().classes("w-full h-full").mark("ifc-viewer")
                    self.ifc_viewer.show(model_url(key, glb.stat().st_mtime_ns), by_node)
                    self.ifc_viewer.set_hidden_classes(self.ifc_hidden)
                    self.ifc_viewer.on("pick", lambda event: self.on_ifc_pick(event.args, key))
                    self.ifc_viewer.on("ready", lambda event: self.on_ifc_ready(event.args))
                    self.ifc_fullscreen_button = ui.button(icon="fullscreen", on_click=self.toggle_ifc_fullscreen).props(
                        "flat round dense color=white").classes("absolute").style(
                        "top:8px;left:8px;z-index:10;background:rgba(9,19,29,.65)").mark("ifc-fullscreen")
                    with self.ifc_fullscreen_button:
                        self.ifc_fullscreen_tooltip = ui.tooltip(t("dxf.result.fullscreen"))
                with ui.column().classes("w-80 gap-2 sam-result-side"):
                    with ui.row().classes("items-center gap-2"):
                        ui.button(t("ifc.viewer.fit"), on_click=lambda: self.ifc_viewer.fit() if self.ifc_viewer else None).props("flat no-caps")
                        ui.button(t("ifc.panel.all"), on_click=lambda: self.set_all_ifc_categories(True)).props("flat no-caps").mark("ifc-all")
                        ui.button(t("ifc.panel.none"), on_click=lambda: self.set_all_ifc_categories(False)).props("flat no-caps").mark("ifc-none")
                    ui.label(t("ifc.panel.types")).classes("text-sm")
                    for name, count in counts.items():
                        with ui.row().classes("items-center gap-2 no-wrap w-full"):
                            self.ifc_checks[name] = ui.checkbox(
                                t(f"ifc.category.{name}"), value=name not in self.ifc_hidden,
                                on_change=lambda event, category=name: self.toggle_ifc_category(category, bool(event.value)),
                            ).mark(f"ifc-class-{name}")
                            ui.label(str(count)).classes("sam-muted text-xs")
                    ui.label(t("ifc.panel.section")).classes("text-sm q-mt-sm")
                    self.ifc_section_label = ui.label(t("ifc.panel.section_off")).classes("sam-muted text-xs")
                    self.ifc_section_slider = ui.slider(
                        min=0.0, max=3.0, step=0.05, value=1.2, on_change=lambda event: self.set_ifc_section(event.value),
                    ).props('label label-value="1.20 m"').mark("ifc-section")
                    self.ifc_section_slider.set_enabled(False)  # greyed out until the section switch is on
                    self.ifc_section_switch = ui.switch(
                        t("ifc.panel.section"), value=False,
                        on_change=lambda event: self.set_ifc_section(self.ifc_section_slider.value),
                    ).mark("ifc-section-on")
                    with ui.row().classes("w-full items-center justify-between no-wrap q-mt-sm"):
                        ui.label(t("ifc.props.title")).classes("text-sm")
                        ui.switch(t("ifc.props.show_all"), value=False,
                                  on_change=lambda event: self.set_ifc_show_all(bool(event.value))).props("dense").classes("text-xs").mark("ifc-props-all")
                    self.ifc_props_box = ui.column().classes("gap-1 w-full").style(
                        "max-height:340px;overflow-y:auto;padding-right:6px").mark("ifc-props")
                    self.ifc_show_all = False
                    self.ifc_picked: str | None = None
                    self.ifc_key = key
                    self.show_ifc_properties(None, key)
            ui.keyboard(on_key=self.on_ifc_key, ignore=["input", "select", "textarea"])

    def toggle_ifc_category(self, category: str, visible: bool) -> None:
        if visible:
            self.ifc_hidden.discard(category)
        else:
            self.ifc_hidden.add(category)
        if self.ifc_viewer is not None:
            self.ifc_viewer.set_hidden_classes(self.ifc_hidden)

    def set_all_ifc_categories(self, visible: bool) -> None:
        self.ifc_hidden = set() if visible else set(self.ifc_checks)
        for box in self.ifc_checks.values():
            box.value = visible
        if self.ifc_viewer is not None:
            self.ifc_viewer.set_hidden_classes(self.ifc_hidden)

    def set_ifc_section(self, height: float | None) -> None:
        """Horizontal section: the slider height while the section switch is on, otherwise no clipping."""
        self.ifc_section_slider.set_enabled(bool(self.ifc_section_switch.value))
        if height is not None:
            height = round(float(height), 2)
            self.ifc_section_slider.props(f'label-value="{height:.2f} m"')
        if not self.ifc_section_switch.value:
            height = None
        if height is None:
            self.ifc_section_label.set_text(t("ifc.panel.section_off"))
        else:
            self.ifc_section_label.set_text(t("ifc.panel.section_height", height=f"{height:.2f}"))
        if self.ifc_viewer is not None:
            self.ifc_viewer.set_clip_height(height)

    def on_ifc_ready(self, args: dict[str, Any]) -> None:
        """Fit the section slider to the model's height range once the GLB is loaded."""
        low = float(args.get("min_height", 0.0))
        high = float(args.get("max_height", low + 3.0))
        if high <= low:
            return
        # numeric props (a "min=0.0" string prop makes Quasar call toFixed on a string)
        self.ifc_section_slider.props["min"] = round(low, 3)
        self.ifc_section_slider.props["max"] = round(high, 3)
        self.ifc_section_slider.update()
        self.ifc_section_slider.value = round(min(high, low + 1.2), 2)

    def on_ifc_pick(self, args: dict[str, Any], key: str) -> None:
        global_id = args.get("global_id")
        self.show_ifc_properties(str(global_id) if global_id else None, key)

    def set_ifc_show_all(self, show_all: bool) -> None:
        self.ifc_show_all = show_all
        self.show_ifc_properties(self.ifc_picked, self.ifc_key)

    def show_ifc_properties(self, global_id: str | None, key: str) -> None:
        """Fill the property panel from props.json: name, class, type, opening details and property sets.

        Only name, class, type, dimensions (Qto_*), SAMcloud_Scan and Pset_*Common are shown by default;
        the "show all" switch adds the internal sets (Raumlot_*, Lining, Panel ...).
        """
        self.ifc_picked = global_id
        box = self.ifc_props_box
        box.clear()
        with box:
            if global_id is None:
                ui.label(t("ifc.props.none")).classes("sam-muted text-xs")
                return
            info = (bim_ui_data.props(self.controller.project_dir, key).get("elements") or {}).get(global_id)
            if not isinstance(info, dict):
                ui.label(t("ifc.props.unknown")).classes("sam-muted text-xs")
                return
            self._property_rows([
                (t("ifc.props.name"), info.get("name")), (t("ifc.props.class"), info.get("ifc_class")),
                (t("ifc.props.type"), info.get("predefined_type")), (t("ifc.props.storey"), info.get("storey")),
                (t("ifc.props.id"), global_id),
            ])
            details = bim_ui_data.opening_details(info)
            if details:
                self._opening_detail_rows(details)
            for pset, values in bim_ui_data.visible_psets(info, self.ifc_show_all):
                ui.label(str(pset)).classes("text-xs q-mt-xs").style("font-weight:600")
                if isinstance(values, dict):
                    self._property_rows(list(values.items()))

    @staticmethod
    def _opening_detail_rows(details: dict[str, Any]) -> None:
        """Window / door summary: width, height, sill (BRH), head (STH), mullions, glazed."""
        ui.label(t("ifc.props.opening")).classes("text-xs q-mt-xs").style("font-weight:600")
        rows: list[tuple[Any, Any]] = []
        for key, label in (("width", "ifc.props.width"), ("height", "ifc.props.height"),
                           ("sill", "ifc.props.sill"), ("head", "ifc.props.head")):
            if key in details:
                rows.append((t(label), t("ifc.props.metres", value=f"{details[key]:.2f}")))
        if "mullions" in details:
            rows.append((t("ifc.props.mullions"), str(details["mullions"])))
        if "glazed" in details:
            rows.append((t("ifc.props.glazed"), t("ifc.props.yes") if details["glazed"] else t("ifc.props.no")))
        if "hinge" in details:
            rows.append((t("ifc.props.hinge"), t(f"ifc.props.hinge_{details['hinge']}")))
        if "swing" in details:
            rows.append((t("ifc.props.swing"), str(details["swing"])))
        if "leaf_angle" in details:
            rows.append((t("ifc.props.leaf_angle"), f"{details['leaf_angle']:.0f} deg"))
        DxfPage._property_rows(rows)

    @staticmethod
    def _property_rows(rows: list[tuple[Any, Any]]) -> None:
        for label, value in rows:
            if value is None or value == "":
                continue
            with ui.row().classes("w-full no-wrap gap-2 items-baseline"):
                ui.label(str(label)).classes("sam-muted text-xs").style("flex:0 0 42%;min-width:0;overflow-wrap:anywhere")
                ui.label(str(value)).classes("text-xs").style("flex:1;min-width:0;overflow-wrap:anywhere")

    def toggle_ifc_fullscreen(self) -> None:
        self.set_ifc_fullscreen(not self.ifc_fullscreen)

    def set_ifc_fullscreen(self, active: bool) -> None:
        row = self.ifc_row
        if row is None:
            return
        self.ifc_fullscreen = active
        if active:
            row.classes(add="sam-result-fullscreen")
        else:
            row.classes(remove="sam-result-fullscreen")
        self.ifc_fullscreen_button.props(f"icon={'fullscreen_exit' if active else 'fullscreen'}")
        self.ifc_fullscreen_tooltip.set_text(t("dxf.result.exit_fullscreen") if active else t("dxf.result.fullscreen"))

    def on_ifc_key(self, event: Any) -> None:
        if self.ifc_fullscreen and event.action.keydown and event.key.escape:
            self.set_ifc_fullscreen(False)

    def build_review_section(self) -> None:
        self.review_section()

    @ui.refreshable_method
    def review_section(self) -> None:
        controller = self.controller
        with ui.card().classes("w-full"):
            ui.label(t("dxf.review.title")).classes("sam-section-title")
            ui.label(t("dxf.review.summary", project=controller.project_name, mode=_mode_label(controller.mode),
                       profile=controller.profile or t("dxf.header.not_set"), scale=controller.scale_status)).classes("sam-muted text-sm")
            with ui.row().classes("items-center gap-3"):
                self.review_level_select()
                self.job_button(t("dxf.action.generate"), "auto_fix_high", lambda: self.start("generate", self.review_level),
                                enabled=controller.mode != "indoor" or self.review_level is not None)
            self.review_status()
            result = controller.plan_result(self.review_level, controller.state.plan_variant) if (controller.mode != "indoor" or self.review_level) else None
            if controller.mode == "indoor" and self.review_level is None:
                ui.label(t("dxf.review.no_levels")).classes("sam-muted")
            elif result is None:
                no_cache = t("dxf.review.no_cache_ortho") if controller.state.plan_variant == "ortho" else t("dxf.review.no_cache_pointcloud")
                ui.label(no_cache).classes("sam-muted").mark("no-cache")
            else:
                self.build_review_canvas(result)
            self.build_review_settings()

    @ui.refreshable_method
    def review_level_select(self) -> None:
        controller = self.controller
        if controller.mode == "indoor":
            options = {str(level["id"]): t("dxf.review.level_option", name=level["name"]) for level in controller.checked_levels()}
            ui.select(options, label=t("dxf.review.level_label"), value=self.review_level if self.review_level in options else None,
                      on_change=lambda event: self.switch_level(event.value)).classes("w-48").props("outlined dense").mark("review-level")
        else:
            ui.label(t("dxf.review.site_plan")).classes("text-sm")

    def switch_level(self, level_id: str | None) -> None:
        if level_id == self.review_level:
            return
        self.review_level = level_id
        self.hidden_layers = set()
        self.review_section.refresh()
        self.apply_busy(self.controller.is_running())

    def build_review_canvas(self, result: Any) -> None:
        controller = self.controller
        view = controller.state.view
        level = self.review_level
        has_rgb = controller.background_path(level, "rgb") is not None
        source = str(view.get("image_source", "rgb"))
        if source == "rgb" and not has_rgb:
            source = "classes"
        ortho = controller.ortho_result(level)
        compare = bool(view.get("compare_plans", False))
        use_ortho = (controller.background_source == "ortho" or compare) and ortho is not None
        image = ortho_preview_url(controller, level) if use_ortho else background_url(controller, level, source)
        bounds = ortho.grid.bounds if use_ortho else result.background.bounds
        variant = controller.state.plan_variant
        primary_variant = "pointcloud" if compare else variant
        compare_variant = "ortho" if compare else None
        primary_result = controller.plan_result(level, primary_variant)
        compare_result = controller.plan_result(level, compare_variant) if compare_variant else None
        if primary_result is None:
            return
        with ui.row().classes("w-full gap-4 no-wrap items-stretch"):
            with ui.element("div").classes("flex-1").style("height:640px"):
                self.review_canvas = DxfReviewCanvas().classes("w-full h-full").mark("review-canvas")
                self.review_canvas.show(
                    payload_url(controller, level, primary_variant),
                    image,
                    bounds,
                    payload_url(controller, level, compare_variant) if compare_result is not None and compare_variant else None,
                )
                primary_opacity = float(view.get("pointcloud_opacity", 1.0)) if compare else float(view.get("dxf_opacity", 1.0))
                self.review_canvas.set_opacity(float(view.get("image_opacity", 1.0)), primary_opacity)
                self.review_canvas.set_compare_opacity(float(view.get("clean_opacity", 1.0)))
                self.review_canvas.set_hidden_layers(sorted(self.hidden_layers))
                self.review_canvas.on("measure", lambda event: self.on_review_measure(event.args, result.scale_factor))
            with ui.column().classes("w-80 gap-2"):
                with ui.row().classes("items-center gap-2"):
                    ui.toggle({"pan": t("dxf.view.pan"), "measure": t("dxf.view.measure")}, value="pan",
                              on_change=lambda event: self.review_canvas.set_tool(event.value) if self.review_canvas else None).mark("review-tool")
                    ui.button(t("dxf.view.fit"), on_click=lambda: self.review_canvas.fit() if self.review_canvas else None).props("flat no-caps")
                ui.toggle({"rgb": "RGB", "classes": t("dxf.view.class_colours")}, value=source,
                          on_change=lambda event: self.set_image_source(event.value)).mark("image-source").set_enabled(has_rgb)
                ui.label(t("dxf.review.image_opacity")).classes("text-sm")
                ui.slider(min=0, max=100, step=1, value=round(float(view.get("image_opacity", 1.0)) * 100),
                          on_change=lambda event: self.set_opacity("image_opacity", event.value)).props("label").mark("image-opacity")
                ui.label(t("dxf.review.dxf_opacity")).classes("text-sm")
                ui.slider(min=0, max=100, step=1, value=round(float(view.get("dxf_opacity", 1.0)) * 100),
                          on_change=lambda event: self.set_opacity("dxf_opacity", event.value)).props("label").mark("dxf-opacity")
                self.measure_label = ui.label("").classes("text-sm")
                ui.label(t("dxf.view.layers")).classes("text-sm q-mt-sm")
                for name, status in result.statuses.items():
                    color = ",".join(str(value) for value in _layer_rgb(result, name))
                    with ui.row().classes("items-center gap-2 no-wrap w-full"):
                        ui.element("div").classes("sam-dxf-swatch").style(f"background:rgb({color})")
                        ui.checkbox(name, value=name not in self.hidden_layers,
                                    on_change=lambda event, layer=name: self.toggle_layer(layer, bool(event.value))).mark(f"layer-{name}")
                        ui.label(str(status.entity_count)).classes("sam-muted text-xs")
                    if status.state != "ok" and status.reason:
                        ui.label(f"{translate_or(f'dxf.layer_state.{status.state}', status.state)}: {status.reason}").classes("sam-muted text-xs q-ml-lg")

    def set_image_source(self, source: str) -> None:
        self.controller.set_view_option("image_source", source)
        if self.review_canvas is not None:
            self.review_canvas.set_image(background_url(self.controller, self.review_level, source))

    def set_opacity(self, key: str, percent: float | None) -> None:
        value = max(0.0, min(1.0, float(percent or 0.0) / 100.0))
        self.controller.set_view_option(key, value)
        view = self.controller.state.view
        if self.review_canvas is not None:
            self.review_canvas.set_opacity(float(view.get("image_opacity", 1.0)), float(view.get("dxf_opacity", 1.0)))

    def set_compare_opacity(self, key: str, percent: float | None, clean: bool) -> None:
        """Persist one compare-overlay opacity and update only that canvas layer."""
        value = max(0.0, min(1.0, float(percent or 0.0) / 100.0))
        self.controller.set_view_option(key, value)
        if self.review_canvas is not None:
            if clean:
                self.review_canvas.set_compare_opacity(value)
            else:
                self.review_canvas.set_opacity(float(self.controller.state.view.get("image_opacity", 1.0)), value)

    def toggle_layer(self, layer: str, visible: bool) -> None:
        if visible:
            self.hidden_layers.discard(layer)
        else:
            self.hidden_layers.add(layer)
        if self.review_canvas is not None:
            self.review_canvas.set_hidden_layers(sorted(self.hidden_layers))

    def on_review_measure(self, args: dict[str, Any], points_scale: float) -> None:
        distance = float((args or {}).get("distance", 0.0))
        if self.measure_label is not None and not self.measure_label.is_deleted:
            self.measure_label.set_text(t("dxf.review.measured", distance=f"{distance:.3f}"))
        self.open_measure_dialog(args or {}, points_scale)

    @ui.refreshable_method
    def review_status(self) -> None:
        controller = self.controller
        level = self.review_level
        if controller.mode == "indoor" and level is None:
            return
        if controller.is_stale(level):
            _banner(t("dxf.review.stale")).mark("review-stale")
        ortho_status = controller.ortho_status(level)
        segment_status = controller.segment_status(level)
        clean_status = controller.plan_status(level, "ortho")
        selected_variant = controller.state.plan_variant
        colours = {"missing": "grey", "ok": "positive", "stale": "warning"}
        with ui.row().classes("items-center gap-3"):
            ui.toggle({"local": t("dxf.review.coords_local"), "original": t("dxf.review.coords_original")},
                      value=str(controller.state.view.get("coordinates", "local")),
                      on_change=lambda event: controller.set_view_option("coordinates", event.value)).mark("coordinates")
            ui.select(
                {"pointcloud": t("dxf.review.variant_pointcloud"), "ortho": t("dxf.review.variant_ortho")},
                label=t("dxf.review.plan_variant"),
                value=selected_variant,
                on_change=lambda event: self.set_plan_variant(event.value),
            ).classes("w-56").props("outlined dense").mark("plan-variant")
            ui.switch(t("dxf.review.compare_plans"), value=bool(controller.state.view.get("compare_plans", False)),
                      on_change=lambda event: self.set_compare_mode(bool(event.value))).mark("compare-plans")
        with ui.row().classes("items-center gap-3 w-full no-wrap"):
            self.job_button(t("dxf.review.build_ortho"), "satellite_alt", lambda: self.start("ortho", level),
                            enabled=controller.mode != "indoor" or level is not None).mark("ortho-build")
            ui.chip(t("dxf.review.chip_ortho", status=_cache_status_label(ortho_status)), color=colours[ortho_status], text_color="white").mark("ortho-status")
            source_options = {"pointcloud": t("dxf.review.source_pointcloud"), "sam3": "SAM3", "combined": t("dxf.review.source_combined")}
            if controller.mode != "indoor":
                source_options["flurseg"] = "FlurSeg"
            ui.select(source_options, label=t("dxf.review.segmentation_model"), value=controller.state.segment_source,
                      on_change=lambda event: self.set_segment_source(event.value)).classes("w-52").props("outlined dense").mark("segment-source")
            self.job_button(t("dxf.review.segment"), "category", lambda: self.start("segment", level),
                            enabled=ortho_status == "ok").mark("segment-build")
            ui.chip(t("dxf.review.chip_segment", status=_cache_status_label(segment_status)), color=colours[segment_status], text_color="white").mark("segment-status")
            self.job_button(t("dxf.review.generate_clean"), "architecture", lambda: self.start("clean_plan", level),
                            enabled=segment_status == "ok").mark("clean-plan-build")
            ui.chip(t("dxf.review.chip_clean", status=_cache_status_label(clean_status)), color=colours[clean_status], text_color="white").mark("clean-plan-status")
            self.job_button(t("dxf.action.export"), "download", lambda: self.start("export", level),
                            enabled=controller.export_enabled(level, selected_variant)).mark("export")
            export_status = controller.plan_status(level, selected_variant)
            ui.chip(t("dxf.review.chip_export", status=_cache_status_label(export_status)), color=colours[export_status], text_color="white").mark("export-status")
            if controller.mode == "indoor":
                self.job_button(t("dxf.review.export_all"), "download_for_offline", lambda: self.start("export", None),
                                enabled=all(controller.export_enabled(target, selected_variant) for target in controller.review_targets())).mark("export-all")
            ui.button(t("dxf.view.save_snapshot"), icon="photo_camera", on_click=self.save_snapshot).props("flat no-caps").set_enabled(controller.cached_result(level) is not None)
        if bool(controller.state.view.get("compare_plans", False)):
            with ui.row().classes("items-center gap-3"):
                ui.label(t("dxf.review.pointcloud_overlay_opacity")).classes("text-sm")
                ui.slider(min=0, max=100, step=1, value=round(float(controller.state.view.get("pointcloud_opacity", 1.0)) * 100),
                          on_change=lambda event: self.set_compare_opacity("pointcloud_opacity", event.value, False)).props("label").mark("compare-pointcloud-opacity")
                ui.label(t("dxf.review.clean_overlay_opacity")).classes("text-sm")
                ui.slider(min=0, max=100, step=1, value=round(float(controller.state.view.get("clean_opacity", 1.0)) * 100),
                          on_change=lambda event: self.set_compare_opacity("clean_opacity", event.value, True)).props("label").mark("compare-clean-opacity")
        with ui.row().classes("items-center gap-3"):
            has_ortho = controller.ortho_result(level) is not None
            ui.toggle({"topview": t("dxf.review.topview"), "ortho": t("dxf.view.orthophoto")}, value=controller.background_source,
                      on_change=lambda event: self.set_background_source(event.value)).mark("ortho-background").set_enabled(has_ortho)
            self.ortho_link(t("dxf.review.link_jpg8000"), "jpg8000", level, "ortho-link-jpg8000")
            self.ortho_link(t("dxf.review.link_jpg4096"), "jpg4096", level, "ortho-link-jpg4096")
            self.ortho_link(t("dxf.review.link_png"), "png", level, "ortho-link-png")
            ui.button(t("dxf.review.open_folder"), icon="folder_open", on_click=lambda: self.open_ortho_folder(level)).props("flat no-caps").mark("ortho-folder").set_enabled(has_ortho)
        last = controller.last_export()
        if last:
            with ui.column().classes("gap-0"):
                coordinates = str(last.get("coordinates", "local"))
                ui.label(t("dxf.review.last_export", coordinates=translate_or(f"dxf.review.coordinate_value.{coordinates}", coordinates),
                           audit=last.get("audit", t("dxf.review.audit_unknown")), detail=last.get("audit_detail", ""))).classes("text-sm").mark("last-export")
                for path in last.get("files", []):
                    ui.label(str(path)).classes("sam-muted text-xs")
        self.apply_busy(controller.is_running())

    def set_segment_source(self, source: str | None) -> None:
        if not source:
            return
        try:
            self.controller.set_segment_source(source)
        except ValueError as error:
            ui.notify(str(error), type="negative")
            return
        self.review_section.refresh()

    def set_plan_variant(self, variant: str | None) -> None:
        if not variant:
            return
        self.controller.set_plan_variant(variant)
        result = self.controller.plan_result(self.review_level, variant)
        if self.review_canvas is None or result is None:
            self.review_section.refresh()
            return
        view = self.controller.state.view
        compare = bool(view.get("compare_plans", False))
        primary_variant = "pointcloud" if compare else variant
        primary = self.controller.plan_result(self.review_level, primary_variant)
        if primary is None:
            self.review_section.refresh()
            return
        ortho = self.controller.ortho_result(self.review_level)
        use_ortho = (self.controller.background_source == "ortho" or compare) and ortho is not None
        image_source = str(view.get("image_source", "rgb"))
        image = ortho_preview_url(self.controller, self.review_level) if use_ortho else background_url(self.controller, self.review_level, image_source)
        bounds = ortho.grid.bounds if use_ortho else primary.background.bounds
        clean = self.controller.plan_result(self.review_level, "ortho") if compare else None
        self.review_canvas.show(
            payload_url(self.controller, self.review_level, primary_variant), image, bounds,
            payload_url(self.controller, self.review_level, "ortho") if clean is not None and compare else None,
        )
        self.review_status.refresh()

    def set_compare_mode(self, enabled: bool) -> None:
        self.controller.set_view_option("compare_plans", enabled)
        self.set_plan_variant(self.controller.state.plan_variant)

    def ortho_link(self, label: str, kind: str, level: str | None, marker: str) -> None:
        """Add an orthophoto download link, disabled until its cached artifact exists."""
        path = self.controller.ortho_file(level, kind)
        if path is None:
            ui.label(label).classes("sam-muted text-sm").mark(marker)
            return
        from samcloud.app.dxf_review_canvas import ORTHO_ROUTE, level_key

        href = f"{ORTHO_ROUTE}/{level_key(level)}/{kind}?v={self.controller.ortho_version(level)}"
        ui.link(label, target=href, new_tab=True).classes("text-sm").mark(marker)

    def set_background_source(self, source: str) -> None:
        self.controller.set_background_source(source)
        if self.review_canvas is not None:
            level = self.review_level
            variant = self.controller.state.plan_variant
            compare = bool(self.controller.state.view.get("compare_plans", False))
            primary_variant = "pointcloud" if compare else variant
            primary = self.controller.plan_result(level, primary_variant)
            ortho = self.controller.ortho_result(level)
            if primary is not None:
                image = ortho_preview_url(self.controller, level) if source == "ortho" and ortho is not None else background_url(
                    self.controller, level, str(self.controller.state.view.get("image_source", "rgb"))
                )
                bounds = ortho.grid.bounds if source == "ortho" and ortho is not None else primary.background.bounds
                clean = self.controller.plan_result(level, "ortho") if compare else None
                self.review_canvas.show(
                    payload_url(self.controller, level, primary_variant), image, bounds,
                    payload_url(self.controller, level, "ortho") if clean is not None and compare else None,
                )
        self.review_status.refresh()

    def open_ortho_folder(self, level: str | None) -> None:
        """Open the existing orthophoto cache folder in Windows Explorer."""
        folder = self.controller.ortho_dir(level)
        if not folder.is_dir():
            ui.notify(t("dxf.review.no_ortho_folder"), type="warning")
            return
        try:
            os.startfile(str(folder))
        except OSError as error:
            ui.notify(t("dxf.review.open_ortho_folder_failed", error=error), type="negative")

    async def save_snapshot(self, variant: str = "pointcloud", level_id: str | None = None) -> None:
        view = self.controller.state.view
        target = self.review_level if level_id is None else level_id
        try:
            path = await run.io_bound(
                self.controller.save_snapshot, target, str(view.get("image_source", "rgb")),
                float(view.get("image_opacity", 1.0)), float(view.get("dxf_opacity", 1.0)), sorted(self.hidden_layers),
                variant, variant == "ortho" and str(view.get("result_background", "ortho")) == "ortho",
            )
        except (ValueError, RuntimeError) as error:
            ui.notify(str(error), type="negative")
            return
        ui.notify(t("dxf.review.snapshot_saved", path=path))

    def build_review_settings(self) -> None:
        controller = self.controller
        view = controller.state.view
        with ui.expansion(t("dxf.settings.title"), icon="tune").classes("w-full"):
            with ui.row().classes("items-end gap-4"):
                ui.number(t("dxf.settings.cell_size"), value=controller.state.overrides.get("cell_size"), min=0.001, format="%.3f",
                          on_change=lambda event: self.change_setting(controller.set_cell_size, event.value if event.value else None)).classes("w-56").props("outlined dense clearable").mark("cell-size")
                if controller.mode == "indoor":
                    ui.toggle({"centerlines": t("dxf.settings.wall_centerlines"), "polygons": t("dxf.settings.wall_polygons")},
                              value="centerlines" if view.get("wall_centerlines", True) else "polygons",
                              on_change=lambda event: self.change_setting(controller.set_view_option, "wall_centerlines", event.value == "centerlines")).mark("wall-output")
                    ui.switch(t("dxf.settings.door_swing"), value=bool(view.get("door_swing_arc", False)),
                               on_change=lambda event: self.change_setting(controller.set_view_option, "door_swing_arc", bool(event.value))).mark("door-swing")
            ui.label(t("dxf.view.orthophoto")).classes("text-sm q-mt-md")
            with ui.row().classes("items-end gap-4"):
                ortho = controller.state.ortho
                ui.number(t("dxf.settings.pixel_outdoor"), value=ortho.get("pixel_size_outdoor"), min=0.001, format="%.3f",
                          on_change=lambda event: self.change_setting(controller.set_ortho_option, "pixel_size_outdoor", event.value if event.value else None)).classes("w-56").props("outlined dense clearable").mark("ortho-pixel-outdoor")
                ui.number(t("dxf.settings.pixel_indoor"), value=ortho.get("pixel_size_indoor"), min=0.001, format="%.3f",
                          on_change=lambda event: self.change_setting(controller.set_ortho_option, "pixel_size_indoor", event.value if event.value else None)).classes("w-56").props("outlined dense clearable").mark("ortho-pixel-indoor")
                ui.number(t("dxf.settings.max_side"), value=ortho.get("max_side"), min=1000, max=30000, step=1,
                          on_change=lambda event: self.change_setting(controller.set_ortho_option, "max_side", event.value if event.value else None)).classes("w-56").props("outlined dense clearable").mark("ortho-max-side")
                ui.number(t("dxf.settings.cut_below_ceiling"), value=ortho.get("cut_below_ceiling"), min=0.0, max=1.0, format="%.3f",
                          on_change=lambda event: self.change_setting(controller.set_ortho_option, "cut_below_ceiling", event.value if event.value else None)).classes("w-56").props("outlined dense clearable").mark("ortho-cut")
            for row in controller.layer_settings():
                name = row["name"]
                band = row["z_band"] or [None, None]
                with ui.row().classes("items-center gap-3 no-wrap"):
                    ui.label(name).classes("w-24 text-sm")
                    ui.select(list(LAYER_MODES), value=row["mode"] if row["mode"] in LAYER_MODES else None,
                              on_change=lambda event, layer=name: self.change_setting(controller.set_layer_mode, layer, event.value) if event.value else None).classes("w-40").props("outlined dense").mark(f"layer-mode-{name}")
                    low = ui.number(t("dxf.settings.z_min"), value=band[0], format="%.2f").classes("w-24").props("outlined dense").mark(f"layer-zmin-{name}")
                    high = ui.number(t("dxf.settings.z_max"), value=band[1], format="%.2f").classes("w-24").props("outlined dense").mark(f"layer-zmax-{name}")
                    low.on_value_change(lambda _event, layer=name, a=low, b=high: self.change_band(layer, a.value, b.value))
                    high.on_value_change(lambda _event, layer=name, a=low, b=high: self.change_band(layer, a.value, b.value))
                    if row["overridden"]:
                        ui.label(t("dxf.settings.project_override")).classes("sam-muted text-xs")

    def change_band(self, layer: str, low: float | None, high: float | None) -> None:
        if low is None or high is None:
            return
        self.change_setting(self.controller.set_layer_z_band, layer, low, high)

    def change_setting(self, action: Callable[..., Any], *args: Any) -> None:
        try:
            action(*args)
        except ValueError as error:
            ui.notify(str(error), type="negative")
            return
        self.settings_changed()


def _layer_rgb(result: Any, name: str) -> tuple[int, int, int]:
    from samcloud.dxf.result import aci_to_rgb

    return aci_to_rgb(int(result.layer_colors.get(name, 7)))
