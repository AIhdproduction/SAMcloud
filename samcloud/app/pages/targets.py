"""Printable ground control target templates (AprilTags and cross targets)."""

from __future__ import annotations

from nicegui import ui

from samcloud.app.layout import render_page
from samcloud.app.state import state
from samcloud.control_points.apriltag_print import build_apriltag_pdf
from samcloud.control_points.apriltags import APRILTAG_ID_MAX
from samcloud.control_points.cross_target_print import (
    CROSS_MAX_COUNT,
    CROSS_MAX_NUMBER,
    build_cross_target_pdf,
)


def _default_apriltag_start() -> int:
    project = state.active_project()
    if project is None:
        return 0
    tag_settings = project.get("settings", {}).get("apriltags", {})
    last_id = tag_settings.get("last_printed_id")
    if last_id is None or int(last_id) < 0:
        return 0
    return int(last_id) + 1


def render() -> None:
    """Render the targets printable templates page."""

    def content() -> None:
        ui.label(
            "Druckvorlagen für Passpunkte. Funktioniert auch ohne geöffnetes Projekt."
        ).classes("sam-muted text-sm q-mb-md")

        with ui.card().classes("w-full q-mb-md"):
            ui.label("AprilTags (150 mm, tag36h11)").classes("text-lg font-medium")
            ui.label(
                "Für Innenräume: Maßstab und optional Georeferenzierung. Mit 100 % drucken."
            ).classes("sam-muted text-sm")
            with ui.row().classes("w-full items-center gap-4 q-mt-sm"):
                tag_count = ui.number(
                    "Anzahl", value=10, min=1, max=100, step=1, precision=0
                ).classes("w-32").mark("targets-apriltag-count")
                tag_start = ui.number(
                    "Start-ID",
                    value=_default_apriltag_start(),
                    min=0,
                    max=APRILTAG_ID_MAX,
                    step=1,
                    precision=0,
                ).classes("w-32").mark("targets-apriltag-start")

                def create_apriltag_pdf() -> None:
                    try:
                        start, count = int(tag_start.value), int(tag_count.value)
                        pdf = build_apriltag_pdf(start, count)
                    except (TypeError, ValueError) as error:
                        ui.notify(str(error), type="negative")
                        return
                    end = start + count - 1
                    ui.download(
                        pdf,
                        filename=f"apriltags_{start}-{end}.pdf",
                        media_type="application/pdf",
                    )
                    project_directory = state.active_project_directory
                    project = state.active_project()
                    if project_directory is not None and project is not None:
                        tag_settings = project.setdefault("settings", {}).setdefault("apriltags", {})
                        tag_settings["last_printed_id"] = end
                        state.projects.save(project_directory, project)
                        tag_start.value = end + 1

                ui.button(
                    "PDF erstellen", on_click=create_apriltag_pdf
                ).classes("sam-button-primary").mark("targets-apriltag-pdf")

        with ui.card().classes("w-full"):
            ui.label("Kreuz-Targets (180 mm, nummeriert)").classes("text-lg font-medium")
            ui.label(
                "Für Drohnenflüge und aussen: zwei schwarze Dreiecke auf weissem Papier, Spitzen = Messpunkt. Ganzes Blatt oder Quadrat an den Marken ausgeschnitten flach auf den Untergrund legen. Funktioniert auf hellem und dunklem Untergrund."
            ).classes("sam-muted text-sm")
            with ui.row().classes("w-full items-center gap-4 q-mt-sm"):
                cross_count = ui.number(
                    "Anzahl",
                    value=10,
                    min=1,
                    max=CROSS_MAX_COUNT,
                    step=1,
                    precision=0,
                ).classes("w-32").mark("targets-cross-count")
                cross_start = ui.number(
                    "Start-Nr.",
                    value=1,
                    min=1,
                    max=CROSS_MAX_NUMBER,
                    step=1,
                    precision=0,
                ).classes("w-32").mark("targets-cross-start")

                def create_cross_pdf() -> None:
                    try:
                        start, count = int(cross_start.value), int(cross_count.value)
                        pdf = build_cross_target_pdf(start, count)
                    except (TypeError, ValueError) as error:
                        ui.notify(str(error), type="negative")
                        return
                    end = start + count - 1
                    ui.download(
                        pdf,
                        filename=f"kreuz_targets_{start}-{end}.pdf",
                        media_type="application/pdf",
                    )

                ui.button(
                    "PDF erstellen", on_click=create_cross_pdf
                ).classes("sam-button-primary").mark("targets-cross-pdf")

    render_page("Targets", content, current_path="/targets")
