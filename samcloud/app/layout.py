"""Shared page frame and navigation."""

from collections.abc import Callable

from nicegui import ui


def render_page(title: str, content: Callable[[], None]) -> None:
    """Render a page with the common SAMcloud navigation shell."""
    with ui.header().classes("items-center justify-between bg-slate-900"):
        ui.label("SAMcloud").classes("text-xl font-bold")
        with ui.row().classes("gap-2"):
            ui.button("Projects", on_click=lambda: ui.navigate.to("/projects")).props("flat")
            ui.button("Sparse Cloud", on_click=lambda: ui.navigate.to("/sparse")).props("flat")
            ui.button("Control Points", on_click=lambda: ui.navigate.to("/control-points")).props("flat")
            ui.button("Dense Cloud", on_click=lambda: ui.navigate.to("/dense")).props("flat")
            ui.button("Settings", on_click=lambda: ui.navigate.to("/settings")).props("flat")
    with ui.column().classes("w-full max-w-7xl mx-auto p-6 gap-5"):
        ui.label(title).classes("text-3xl font-semibold")
        content()
