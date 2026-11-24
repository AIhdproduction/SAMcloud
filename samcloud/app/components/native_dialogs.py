"""Native (Tk) file and folder pickers for the local desktop application."""

from __future__ import annotations

from pathlib import Path
from tkinter import TclError, Tk, filedialog

from nicegui import ui

from samcloud.app.i18n import t


def _initial_directory(initial: str, *, is_file: bool) -> str | None:
    """Existing directory to open the dialog in (the parent folder for a file path), else None."""
    try:
        if not (initial or "").strip():
            return None
        location = Path(initial.strip())
        candidate = location.parent if is_file and not location.is_dir() else location
        return str(candidate) if candidate.is_dir() else None
    except (OSError, ValueError):
        return None


def _open_dialog(show) -> str | None:  # noqa: ANN001 - a callable taking the hidden Tk root
    root = None
    try:
        root = Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        selected = show(root)
    except TclError as error:
        ui.notify(t("projects.create.dialog_failed", error=error), type="negative")
        return None
    finally:
        if root is not None:
            root.destroy()
    return str(selected) if selected else None


def choose_directory(initial: str, title: str) -> str | None:
    """Ask for a folder; None when cancelled or when the dialog cannot be opened."""
    return _open_dialog(lambda root: filedialog.askdirectory(
        initialdir=_initial_directory(initial, is_file=False), parent=root, title=title, mustexist=True,
    ))


def choose_file(initial: str, title: str, filetypes: list[tuple[str, str]]) -> str | None:
    """Ask for a file; None when cancelled or when the dialog cannot be opened."""
    return _open_dialog(lambda root: filedialog.askopenfilename(
        initialdir=_initial_directory(initial, is_file=True), parent=root, title=title, filetypes=filetypes,
    ))


def _append_slot(field: ui.input):
    """The field's append slot; an existing one is reused because add_slot would replace it (and its buttons)."""
    return field.slots.get("append") or field.add_slot("append")


def attach_folder_button(field: ui.input, title: str, icon: str = "folder_open") -> None:
    """Append a button to ``field`` that fills it with a folder chosen in the native dialog."""
    def pick() -> None:
        selected = choose_directory(field.value or "", title)
        if selected:
            field.value = selected

    with _append_slot(field):
        ui.button(icon=icon, on_click=pick).props(f"flat round dense aria-label='{title}'")


def attach_file_button(field: ui.input, title: str, filetypes: list[tuple[str, str]], icon: str = "movie") -> None:
    """Append a button to ``field`` that fills it with a file chosen in the native dialog."""
    def pick() -> None:
        selected = choose_file(field.value or "", title, filetypes)
        if selected:
            field.value = selected

    with _append_slot(field):
        ui.button(icon=icon, on_click=pick).props(f"flat round dense aria-label='{title}'")
