"""Card "Protokoll" on the control-points page: create the processing report PDF and download earlier ones."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

from nicegui import ui

from samcloud.app.i18n import current_language, t
from samcloud.report.availability import protocol_availability
from samcloud.report.job import protocol_manager

_MISSING_KEYS = {"control_points": "control_points.protocol.missing_control_points",
                 "step6": "control_points.protocol.missing_step6"}
_PHASE_KEYS = {"collect": "control_points.protocol.phase_collect", "charts": "control_points.protocol.phase_charts",
               "pdf": "control_points.protocol.phase_pdf"}


def _config_sections() -> dict[str, bool]:
    try:
        from samcloud.report.config import load_protocol_config

        sections = load_protocol_config().get("sections", {})
        return {"class_details": bool(sections.get("class_details", True)), "camera_list": bool(sections.get("camera_list", True))}
    except Exception:
        return {"class_details": True, "camera_list": True}


def _list_protocols(project_dir: Path) -> list[Path]:
    try:
        from samcloud.report.generate import list_protocols

        return list(list_protocols(project_dir))
    except ImportError:
        return sorted((project_dir / "reports").glob("protocol_*.pdf"), key=lambda p: p.name, reverse=True)


def _download(path: Path) -> None:
    try:
        ui.download(path.read_bytes(), filename=path.name, media_type="application/pdf")
    except OSError as error:
        ui.notify(str(error), type="negative")


def render_protocol_panel(project_dir: Path, project: dict[str, Any] | None = None) -> None:
    """Render the card; call inside the page content."""
    options = _config_sections()
    ctx: dict[str, Any] = {"shown": None, "notified": None}

    @ui.refreshable
    def panel() -> None:
        availability = protocol_availability(project_dir, project)
        status = protocol_manager.status(project_dir)
        state = status.get("state", "idle")
        running = state == "running"
        ctx["shown"] = (state, bool(status.get("result_path")), availability.enabled)
        with ui.card().classes("w-full q-mt-lg").mark("protocol-card"):
            ui.label(t("control_points.protocol.title")).classes("text-lg font-medium")
            ui.label(t("control_points.protocol.description")).classes("sam-muted text-sm")
            if availability.without_control_points:
                ui.label(t("control_points.protocol.without_control_points")).classes("text-sm q-mt-sm").mark("protocol-hint")
            ui.checkbox(t("control_points.protocol.class_details"), value=options["class_details"],
                        on_change=lambda e: options.__setitem__("class_details", bool(e.value))).mark("protocol-class-details")
            ui.checkbox(t("control_points.protocol.camera_list"), value=options["camera_list"],
                        on_change=lambda e: options.__setitem__("camera_list", bool(e.value))).mark("protocol-camera-list")

            button = ui.button(t("control_points.protocol.create"), icon="picture_as_pdf", on_click=lambda: start()) \
                .props("outline no-caps").classes("q-mt-sm").mark("protocol-button")
            if not availability.enabled:
                button.props("disable")
                with button:
                    ui.tooltip("; ".join(t(_MISSING_KEYS[key]) for key in availability.missing if key in _MISSING_KEYS))
            elif running:
                button.props("disable")

            if running:
                with ui.column().classes("w-full gap-1 q-mt-sm").mark("protocol-progress"):
                    ui.linear_progress(value=float(status.get("fraction") or 0.0), show_value=False).classes("w-full")
                    phase = str(status.get("phase") or "")
                    ui.label(t(_PHASE_KEYS[phase]) if phase in _PHASE_KEYS else t("control_points.protocol.working")).classes("sam-muted text-sm")
            elif state == "error":
                ui.label(t("control_points.protocol.error", error=str(status.get("error") or ""))).classes("text-red-4 text-sm q-mt-sm").mark("protocol-error")
            elif state == "done" and status.get("result_path"):
                result = Path(str(status["result_path"]))
                with ui.row().classes("items-center gap-2 q-mt-sm").mark("protocol-done"):
                    ui.label(t("control_points.protocol.done", name=result.name)).classes("text-sm")
                    ui.button(t("control_points.protocol.download"), icon="download",
                              on_click=lambda path=result: _download(path)).props("flat dense no-caps")

            ui.label(t("control_points.protocol.existing")).classes("font-medium q-mt-md")
            files = _list_protocols(project_dir)
            with ui.column().classes("w-full gap-1").mark("protocol-list"):
                if not files:
                    ui.label(t("control_points.protocol.none")).classes("sam-muted text-sm")
                for path in files:
                    try:
                        info = path.stat()
                        detail = f"{datetime.fromtimestamp(info.st_mtime):%Y-%m-%d %H:%M} · {info.st_size / 1e6:.1f} MB"
                    except OSError:
                        detail = ""
                    with ui.row().classes("items-center gap-2"):
                        ui.label(path.name).classes("text-sm")
                        ui.label(detail).classes("sam-muted text-sm")
                        ui.button(icon="download", on_click=lambda p=path: _download(p)).props("flat dense round").mark("protocol-download")

    def start() -> None:
        try:
            protocol_manager.start(project_dir, language=current_language(), options=dict(options))
        except Exception as error:
            ui.notify(str(error), type="negative")
            return
        poll.activate()
        panel.refresh()

    def tick() -> None:
        status = protocol_manager.status(project_dir)
        state = status.get("state", "idle")
        if state == "running":
            panel.refresh()  # progress bar and phase label change continuously
            return
        poll.deactivate()
        key = (state, str(status.get("result_path")))
        if state in ("done", "error", "cancelled") and ctx["notified"] != key:
            ctx["notified"] = key
            if state == "done":
                ui.notify(t("control_points.protocol.done_notify"), type="positive")
        if ctx["shown"] is None or ctx["shown"][0] != state:
            panel.refresh()

    panel()
    poll = ui.timer(1.0, tick, active=protocol_manager.is_running(project_dir))
