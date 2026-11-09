"""DXF export page: header, preparation, side view (indoor) and DXF review.

Nothing heavy runs automatically: every preview, alignment, level detection, generation and export
is started by an explicit button through DxfPageController.start_job. Settings only mark caches stale.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from nicegui import run, ui

from samcloud.app.dxf_controller import LAYER_MODES, MODEL_PROFILES, DxfPageController, dxf_unavailable_reason
from samcloud.app.dxf_review_canvas import DxfReviewCanvas, background_url, payload_url
from samcloud.app.dxf_side_canvas import DxfSideCanvas
from samcloud.app.dxf_side_view import encode_png_data_url, render_side_view
from samcloud.app.layout import render_page
from samcloud.app.state import state
from samcloud.dxf.job_runner import DxfJobRunner

ACTION_LABELS: dict[str, str] = {
    "preview": "Create preview",
    "align": "Align to gravity",
    "levels": "Detect levels",
    "generate": "Generate DXF preview",
    "export": "Export DXF",
}

_RUNNERS: dict[Path, DxfJobRunner] = {}

_STYLE = """
<style>
.sam-dxf-banner { padding:10px 14px; border-radius:7px; font-size:14px; }
.sam-dxf-banner.is-warning { background:rgba(255,183,77,.13); border:1px solid #a8752b; color:#ffd59a; }
.sam-dxf-banner.is-error { background:rgba(239,107,115,.13); border:1px solid #9b3c43; color:#ffc2c6; }
.sam-dxf-banner.is-info { background:rgba(38,183,239,.10); border:1px solid #24719b; color:#bfe9fb; }
.sam-dxf-swatch { width:14px; height:14px; border-radius:3px; border:1px solid rgba(255,255,255,.35); flex:none; }
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
            ui.label("Open a project before exporting DXF plans.").classes("sam-muted")
            return
        if dxf_unavailable_reason(project_directory) is not None:
            expected = Path(project_directory) / "products" / "classified" / "classified.las"
            ui.label(
                "DXF export needs a classified LAS file, but this project has none yet. "
                f"Export a classified LAS first (expected at {expected}), then open this page again."
            ).classes("sam-muted").mark("dxf-not-ready")
            return
        controller = DxfPageController(project_directory, runner=runner_for(project_directory))
        DxfPage(controller).build()

    render_page("DXF export", content, current_path="/dxf", wide=True)


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
        self.side_canvas: DxfSideCanvas | None = None
        self.side_scale: float = controller.scale_factor
        self.side_rendered = False
        self.review_canvas: DxfReviewCanvas | None = None
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
        self.job_stage = self.job_progress = self.cancel_button = self.job_log = None
        self.build_header()
        if self.controller.profile_required:
            self.build_profile_required()
            return
        blocked = self.controller.blocked_message
        if blocked is not None:
            _banner(blocked, "error").props("data-testid=blocked")
            return
        self.build_job_bar()
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
                ui.label(f"Mode: {controller.mode}").classes("sam-muted").props("data-testid=mode")
                ui.label(f"Model profile: {controller.profile or 'not set'}").classes("sam-muted")
                ui.label(f"Scale: {controller.scale_status}").classes("sam-muted")
            with ui.row().classes("items-end w-full gap-3"):
                current = controller.las_path
                las_input = ui.input("LAS file", value=str(current) if current else "").classes("flex-1").props("outlined dense").mark("las-path")
                ui.button("Use this LAS", icon="check", on_click=lambda: self.choose_las(las_input.value)).props("flat no-caps")
                if controller.las_is_custom:
                    ui.button("Project LAS", icon="restart_alt", on_click=lambda: self.choose_las(None)).props("flat no-caps")
                mode_options = {"auto": "Automatic", "indoor": "Indoor", "outdoor": "Outdoor"}
                ui.select(mode_options, label="Mode", value=controller.state.mode_override or "auto",
                          on_change=lambda event: self.set_mode(event.value)).classes("w-40").props("outlined dense").mark("mode-select")
                if controller.state.profile_name and not controller.profile_required:
                    ui.select(list(MODEL_PROFILES), label="Model profile", value=controller.state.profile_name,
                              on_change=lambda event: self.set_profile(event.value)).classes("w-48").props("outlined dense").mark("profile-change")
            if current is None or not current.is_file():
                _banner("No LAS file found. Export a classified LAS first or choose an existing .las file.", "warning")

    def build_profile_required(self) -> None:
        with ui.card().classes("w-full"):
            _banner("This LAS file carries no model profile. Choose the model that classified it before continuing.", "info")
            ui.select(list(MODEL_PROFILES), label="Model profile (required)", value=None,
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
        with ui.card().classes("w-full"):
            with ui.row().classes("items-center w-full gap-3 no-wrap"):
                self.job_stage = ui.label("").classes("text-sm").mark("job-stage")
                self.job_progress = ui.linear_progress(value=0.0, show_value=False).classes("flex-1")
                self.cancel_button = ui.button("Cancel", icon="stop", on_click=self.cancel).props("flat no-caps color=negative")
            self.job_log = ui.label("").classes("sam-dxf-log w-full")
        self.update_job_bar(status, self.controller.is_running())

    def update_job_bar(self, status: dict[str, Any], running: bool) -> None:
        if self.job_stage is None or self.job_progress is None or self.job_log is None:
            return
        action = ACTION_LABELS.get(str(status.get("action", "")), str(status.get("action", "")) or "Job")
        job_state = str(status.get("state", "idle"))
        if running:
            self.job_stage.set_text(f"{action}: {status.get('stage', '')} ({float(status.get('fraction', 0.0)) * 100:.0f} %)")
        elif job_state == "idle":
            self.job_stage.set_text("No DXF job has run yet.")
        else:
            message = f" - {status['message']}" if status.get("message") else ""
            self.job_stage.set_text(f"Last job: {action} {job_state}{message}")
        self.job_progress.set_value(float(status.get("fraction", 0.0)) if running or job_state == "done" else 0.0)
        show_log = not running and job_state in ("failed", "cancelled", "unknown")
        self.job_log.set_text("\n".join(self.controller.log_tail(25)) if show_log else "")
        self.job_log.set_visibility(show_log)
        if self.cancel_button is not None:
            self.cancel_button.set_visibility(running)

    def start(self, action: str, level_id: str | None = None) -> None:
        try:
            self.controller.start_job(action, level_id)
        except (ValueError, RuntimeError) as error:
            ui.notify(str(error), type="negative")
            return
        self.was_running = True
        self.apply_busy(True)
        self.update_job_bar(self.controller.job_status(), True)
        ui.notify(f"{ACTION_LABELS.get(action, action)} started")

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
            job_state = str(status.get("state", ""))
            action = ACTION_LABELS.get(str(status.get("action", "")), "Job")
            kind = "positive" if job_state == "done" else ("warning" if job_state == "cancelled" else "negative")
            ui.notify(f"{action} {job_state or 'finished'}", type=kind)
            targets = self.controller.review_targets()
            if self.review_level not in targets:
                self.review_level = targets[0] if targets else None
            self.body.refresh()
            return
        self.was_running = running
        self.update_job_bar(status, running)
        self.apply_busy(running)

    # ------------------------------------------------------------------ stale + preparation

    @ui.refreshable_method
    def stale_banner(self) -> None:
        stale = self.controller.stale_levels()
        if stale:
            _banner(f"Settings changed: DXF preview is out of date ({', '.join(stale)}). Regenerate in the DXF review.").mark("stale-banner")

    def settings_changed(self) -> None:
        """Refresh only the staleness indicators; never regenerates anything."""
        self.stale_banner.refresh()
        self.review_status.refresh()

    def build_preparation(self) -> None:
        controller = self.controller
        indoor = controller.mode == "indoor"
        with ui.card().classes("w-full"):
            ui.label("Preparation").classes("sam-section-title")
            with ui.row().classes("items-center gap-3"):
                self.job_button("Create preview", "grain", lambda: self.start("preview"))
                if indoor:
                    self.job_button("Align to gravity", "straighten", lambda: self.start("align"))
                    self.job_button("Detect levels", "layers", lambda: self.start("levels"))
            ui.label("Preview: available" if controller.has_preview else "Preview: not created yet").classes("sam-muted text-sm")
            for warning in controller.plausibility_warnings:
                _banner(warning, "warning")
            with ui.row().classes("items-end gap-3"):
                scale_input = ui.number("Scale factor", value=controller.scale_factor, min=0.0, format="%.6g").classes("w-40").props("outlined dense").mark("scale-factor")
                ui.button("Apply", on_click=lambda: self.apply_scale(scale_input.value)).props("flat no-caps").mark("scale-apply")
                ui.label(f"{controller.scale_source}: {controller.scale_status}").classes("sam-muted text-sm")
            ui.label("Or use the Measure tool in the side view or the DXF review and type the real distance.").classes("sam-muted text-xs")
            if indoor:
                alignment = controller.alignment
                if alignment is None:
                    ui.label("Alignment: not estimated (local Z is used as up).").classes("sam-muted text-sm")
                else:
                    with ui.row().classes("items-center gap-4"):
                        ui.label(f"Alignment: tilt {float(alignment.get('tilt_degrees', 0.0)):.2f}°, "
                                 f"yaw {float(alignment.get('yaw_degrees', 0.0)):.2f}° ({alignment.get('method', '')})").classes("text-sm")
                        ui.switch("Align walls to X/Y", value=controller.align_walls,
                                  on_change=lambda event: self.set_align_walls(event.value)).mark("align-walls")

    def apply_scale(self, value: float | None) -> None:
        try:
            self.controller.set_scale_factor(float(value) if value is not None else float("nan"))
        except ValueError as error:
            ui.notify(str(error), type="negative")
            return
        ui.notify("Scale applied. Cached DXF previews are now out of date.")
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
            ui.label("Side view and levels").classes("sam-section-title")
            with ui.row().classes("items-end gap-3 w-full"):
                ui.toggle({"x": "View along X", "y": "View along Y"}, value=view.get("side_axis", "x"),
                          on_change=lambda event: self.set_side_option("side_axis", event.value)).mark("side-axis")
                ui.number("Slab centre (m)", value=view.get("slab_center"),
                          on_change=lambda event: self.set_side_option("slab_center", event.value)).classes("w-36").props("outlined dense clearable").mark("slab-center")
                ui.number("Slab thickness (m)", value=view.get("slab_thickness"), min=0.0,
                          on_change=lambda event: self.set_side_option("slab_thickness", event.value)).classes("w-36").props("outlined dense clearable").mark("slab-thickness")
                ui.button("Render side view", icon="visibility", on_click=self.render_side).props("flat no-caps").set_enabled(controller.has_preview)
                ui.toggle({"pan": "Pan / drag lines", "add": "Add level", "measure": "Measure"}, value="pan",
                          on_change=lambda event: self.side_canvas.set_tool(event.value) if self.side_canvas else None).mark("side-tool")
                ui.button("Fit", on_click=lambda: self.side_canvas.fit() if self.side_canvas else None).props("flat no-caps")
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
            ui.label("No levels yet. Press Detect levels or add one by clicking in the side view.").classes("sam-muted text-sm")
            return
        ui.label("Checked levels are exported (1 = lowest).").classes("sam-muted text-xs")
        for level in sorted(levels, key=lambda item: float(item["floor_z"])):
            level_id = str(level["id"])
            with ui.row().classes("items-center gap-1 no-wrap w-full"):
                ui.checkbox(value=bool(level.get("checked", True)),
                            on_change=lambda event, lid=level_id: self.level_edit(self.controller.set_level_checked, lid, bool(event.value))).mark(f"level-check-{level_id}")
                name_input = ui.input(value=str(level["name"])).classes("w-16").props("dense outlined").mark(f"level-name-{level_id}")
                name_input.on("blur", lambda _event, lid=level_id, field=name_input: self.rename(lid, field.value))
                name_input.on("keydown.enter", lambda _event, lid=level_id, field=name_input: self.rename(lid, field.value))
                ui.number(value=round(float(level["floor_z"]), 3), format="%.3f",
                          on_change=lambda event, lid=level_id: self.level_edit(self.controller.move_level, lid, floor_z=event.value) if event.value is not None else None).classes("w-24").props("dense outlined label=Floor").mark(f"level-floor-{level_id}")
                ui.number(value=None if level.get("ceiling_z") is None else round(float(level["ceiling_z"]), 3), format="%.3f",
                          on_change=lambda event, lid=level_id: self.level_edit(self.controller.move_level, lid, ceiling_z=event.value) if event.value is not None else None).classes("w-24").props("dense outlined label=Ceiling").mark(f"level-ceiling-{level_id}")
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
            ui.notify("Create the preview first.", type="warning")
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
            ui.label(f"Measured distance: {measured:.4f} m").classes("text-md")
            real = ui.number("Real distance (m)", value=None, min=0.0).props("outlined dense autofocus").mark("real-distance")

            def apply() -> None:
                try:
                    factor = self.controller.measure_scale(args["p1"], args["p2"], float(real.value or 0.0),
                                                           points_scale=points_scale if points_scale is not None else self.side_scale)
                except (ValueError, KeyError, TypeError) as error:
                    ui.notify(str(error), type="negative")
                    return
                dialog.close()
                ui.notify(f"Scale factor {factor:.6g} applied. Cached DXF previews are now out of date.")
                self.body.refresh()

            with ui.row():
                ui.button("Use for scale", on_click=apply).props("unelevated no-caps").mark("apply-measure")
                ui.button("Close", on_click=dialog.close).props("flat no-caps")
        dialog.open()

    # ------------------------------------------------------------------ review

    def build_review_section(self) -> None:
        self.review_section()

    @ui.refreshable_method
    def review_section(self) -> None:
        controller = self.controller
        with ui.card().classes("w-full"):
            ui.label("DXF review").classes("sam-section-title")
            ui.label(f"{controller.project_name} · {controller.mode} · profile {controller.profile or 'not set'} · scale: {controller.scale_status}").classes("sam-muted text-sm")
            with ui.row().classes("items-center gap-3"):
                self.review_level_select()
                self.job_button("Generate DXF preview", "auto_fix_high", lambda: self.start("generate", self.review_level),
                                enabled=controller.mode != "indoor" or self.review_level is not None)
            self.review_status()
            result = controller.cached_result(self.review_level) if (controller.mode != "indoor" or self.review_level) else None
            if controller.mode == "indoor" and self.review_level is None:
                ui.label("Detect or add levels and check at least one to review it.").classes("sam-muted")
            elif result is None:
                ui.label("No DXF preview is cached for this target. Press Generate DXF preview.").classes("sam-muted").mark("no-cache")
            else:
                self.build_review_canvas(result)
            self.build_review_settings()

    @ui.refreshable_method
    def review_level_select(self) -> None:
        controller = self.controller
        if controller.mode == "indoor":
            options = {str(level["id"]): f"Level {level['name']}" for level in controller.checked_levels()}
            ui.select(options, label="Level", value=self.review_level if self.review_level in options else None,
                      on_change=lambda event: self.switch_level(event.value)).classes("w-48").props("outlined dense").mark("review-level")
        else:
            ui.label("Site plan").classes("text-sm")

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
        with ui.row().classes("w-full gap-4 no-wrap items-stretch"):
            with ui.element("div").classes("flex-1").style("height:640px"):
                self.review_canvas = DxfReviewCanvas().classes("w-full h-full").mark("review-canvas")
                self.review_canvas.show(payload_url(controller, level), background_url(controller, level, source), result.background.bounds)
                self.review_canvas.set_opacity(float(view.get("image_opacity", 1.0)), float(view.get("dxf_opacity", 1.0)))
                self.review_canvas.set_hidden_layers(sorted(self.hidden_layers))
                self.review_canvas.on("measure", lambda event: self.on_review_measure(event.args, result.scale_factor))
            with ui.column().classes("w-80 gap-2"):
                with ui.row().classes("items-center gap-2"):
                    ui.toggle({"pan": "Pan", "measure": "Measure"}, value="pan",
                              on_change=lambda event: self.review_canvas.set_tool(event.value) if self.review_canvas else None).mark("review-tool")
                    ui.button("Fit", on_click=lambda: self.review_canvas.fit() if self.review_canvas else None).props("flat no-caps")
                ui.toggle({"rgb": "RGB", "classes": "Class colours"}, value=source,
                          on_change=lambda event: self.set_image_source(event.value)).mark("image-source").set_enabled(has_rgb)
                ui.label("Image opacity").classes("text-sm")
                ui.slider(min=0, max=100, step=1, value=round(float(view.get("image_opacity", 1.0)) * 100),
                          on_change=lambda event: self.set_opacity("image_opacity", event.value)).props("label").mark("image-opacity")
                ui.label("DXF opacity").classes("text-sm")
                ui.slider(min=0, max=100, step=1, value=round(float(view.get("dxf_opacity", 1.0)) * 100),
                          on_change=lambda event: self.set_opacity("dxf_opacity", event.value)).props("label").mark("dxf-opacity")
                self.measure_label = ui.label("").classes("text-sm")
                ui.label("Layers").classes("text-sm q-mt-sm")
                for name, status in result.statuses.items():
                    color = ",".join(str(value) for value in _layer_rgb(result, name))
                    with ui.row().classes("items-center gap-2 no-wrap w-full"):
                        ui.element("div").classes("sam-dxf-swatch").style(f"background:rgb({color})")
                        ui.checkbox(name, value=name not in self.hidden_layers,
                                    on_change=lambda event, layer=name: self.toggle_layer(layer, bool(event.value))).mark(f"layer-{name}")
                        ui.label(str(status.entity_count)).classes("sam-muted text-xs")
                    if status.state != "ok" and status.reason:
                        ui.label(f"{status.state}: {status.reason}").classes("sam-muted text-xs q-ml-lg")

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
            self.measure_label.set_text(f"Measured: {distance:.3f} m")
        self.open_measure_dialog(args or {}, points_scale)

    @ui.refreshable_method
    def review_status(self) -> None:
        controller = self.controller
        level = self.review_level
        if controller.mode == "indoor" and level is None:
            return
        if controller.is_stale(level):
            _banner("Settings changed since this preview was generated. Press Generate DXF preview to update.").mark("review-stale")
        with ui.row().classes("items-center gap-3"):
            ui.toggle({"local": "Local coordinates", "original": "Original coordinates"},
                      value=str(controller.state.view.get("coordinates", "local")),
                      on_change=lambda event: controller.set_view_option("coordinates", event.value)).mark("coordinates")
            self.job_button("Export DXF", "download", lambda: self.start("export", level), enabled=controller.export_enabled(level)).mark("export")
            if controller.mode == "indoor":
                self.job_button("Export all checked levels", "download_for_offline", lambda: self.start("export", None),
                                enabled=controller.all_exports_enabled()).mark("export-all")
            ui.button("Save snapshot PNG", icon="photo_camera", on_click=self.save_snapshot).props("flat no-caps").set_enabled(controller.cached_result(level) is not None)
        last = controller.last_export()
        if last:
            with ui.column().classes("gap-0"):
                ui.label(f"Last export ({last.get('coordinates', 'local')} coordinates), audit {last.get('audit', 'unknown')}: {last.get('audit_detail', '')}").classes("text-sm").mark("last-export")
                for path in last.get("files", []):
                    ui.label(str(path)).classes("sam-muted text-xs")
        self.apply_busy(controller.is_running())

    async def save_snapshot(self) -> None:
        view = self.controller.state.view
        try:
            path = await run.io_bound(
                self.controller.save_snapshot, self.review_level, str(view.get("image_source", "rgb")),
                float(view.get("image_opacity", 1.0)), float(view.get("dxf_opacity", 1.0)), sorted(self.hidden_layers),
            )
        except (ValueError, RuntimeError) as error:
            ui.notify(str(error), type="negative")
            return
        ui.notify(f"Snapshot saved: {path}")

    def build_review_settings(self) -> None:
        controller = self.controller
        view = controller.state.view
        with ui.expansion("Generation settings (changes mark the preview stale; nothing is regenerated)", icon="tune").classes("w-full"):
            with ui.row().classes("items-end gap-4"):
                ui.number("Cell size (m, empty = default)", value=controller.state.overrides.get("cell_size"), min=0.001, format="%.3f",
                          on_change=lambda event: self.change_setting(controller.set_cell_size, event.value if event.value else None)).classes("w-56").props("outlined dense clearable").mark("cell-size")
                if controller.mode == "indoor":
                    ui.toggle({"centerlines": "Wall centre-lines", "polygons": "Wall polygons"},
                              value="centerlines" if view.get("wall_centerlines", True) else "polygons",
                              on_change=lambda event: self.change_setting(controller.set_view_option, "wall_centerlines", event.value == "centerlines")).mark("wall-output")
                    ui.switch("Door swing arc", value=bool(view.get("door_swing_arc", False)),
                              on_change=lambda event: self.change_setting(controller.set_view_option, "door_swing_arc", bool(event.value))).mark("door-swing")
            for row in controller.layer_settings():
                name = row["name"]
                band = row["z_band"] or [None, None]
                with ui.row().classes("items-center gap-3 no-wrap"):
                    ui.label(name).classes("w-24 text-sm")
                    ui.select(list(LAYER_MODES), value=row["mode"] if row["mode"] in LAYER_MODES else None,
                              on_change=lambda event, layer=name: self.change_setting(controller.set_layer_mode, layer, event.value) if event.value else None).classes("w-40").props("outlined dense").mark(f"layer-mode-{name}")
                    low = ui.number("z min", value=band[0], format="%.2f").classes("w-24").props("outlined dense").mark(f"layer-zmin-{name}")
                    high = ui.number("z max", value=band[1], format="%.2f").classes("w-24").props("outlined dense").mark(f"layer-zmax-{name}")
                    low.on_value_change(lambda _event, layer=name, a=low, b=high: self.change_band(layer, a.value, b.value))
                    high.on_value_change(lambda _event, layer=name, a=low, b=high: self.change_band(layer, a.value, b.value))
                    if row["overridden"]:
                        ui.label("project override").classes("sam-muted text-xs")

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
