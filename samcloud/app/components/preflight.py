"""Image count and storage check panel for the project-creation form.

The panel only informs, it never blocks "create". Probing runs in a worker thread after a debounce; every
input change starts a new run and a result of an older run is dropped (``version`` guard).
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from nicegui import background_tasks, run, ui

from samcloud.acquisition import preflight
from samcloud.app.i18n import current_language, t

DEBOUNCE_SECONDS = 0.5
_WARNING_NOTES = {"video_not_mp4", "video_not_2to1", "photos_not_2to1"}


@dataclass(frozen=True)
class PreflightInputs:
    capture_type: str
    image_directory: str
    panorama_video_path: str
    panorama_image_directory: str
    interval: float
    storage: tuple[tuple[str, bool], ...]
    projects_directory: str


def compute(inputs: PreflightInputs, cache: dict) -> preflight.PreflightResult:
    """Blocking part (disk access); the source probe is reused while the paths stay the same."""
    key = (inputs.capture_type, inputs.image_directory, inputs.panorama_video_path, inputs.panorama_image_directory)
    probe = cache.get("probe") if cache.get("key") == key else None
    if probe is None:
        probe = preflight.probe_sources(
            inputs.capture_type, inputs.image_directory, inputs.panorama_video_path, inputs.panorama_image_directory,
        )
        cache["key"], cache["probe"] = key, probe
    free, drive = preflight.free_space(inputs.projects_directory)
    try:
        from samcloud.core.configuration import load_global_config

        requested = int(
            load_global_config("defaults.yml").get("acquisition", {}).get("panorama", {}).get("cubemap_face_size", 1232)
        )
    except Exception:  # noqa: BLE001
        requested = 1232
    return preflight.estimate(
        probe, interval=inputs.interval, storage=dict(inputs.storage), free_bytes=free, free_drive=drive,
        dense_config=preflight.load_dense_config(), requested_face_size=requested,
    )


class PreflightPanel:
    """One panel under the source fields; recomputes on every relevant input change."""

    def __init__(
        self,
        *,
        capture_type: Any,
        image_directory: Any,
        panorama_video_path: Any,
        panorama_image_directory: Any,
        frame_interval: Any,
        storage_switches: dict[str, Any],
        projects_directory: Callable[[], Path | str],
    ) -> None:
        self._capture_type = capture_type
        self._image_directory = image_directory
        self._video = panorama_video_path
        self._photos = panorama_image_directory
        self._interval = frame_interval
        self._storage = storage_switches
        self._projects_directory = projects_directory
        self.version = 0
        self._cache: dict = {}
        self._build()
        for control in (capture_type, image_directory, panorama_video_path, panorama_image_directory, frame_interval,
                        *storage_switches.values()):
            control.on_value_change(lambda _event: self.schedule())
        self._render_empty()

    # ------------------------------------------------------------------------------------------ layout

    def _build(self) -> None:
        self.root = (
            ui.column().classes("w-full gap-1 q-pa-sm rounded-borders").style("min-height: 12rem")
            .props('aria-live=polite role=status').mark("preflight-panel")
        )
        with self.root:
            with ui.row().classes("items-center gap-2"):
                ui.label(t("projects.preflight.title")).classes("text-sm sam-muted")
                self.checking = ui.row().classes("items-center gap-1").mark("preflight-checking")
                with self.checking:
                    ui.spinner(size="xs")
                    ui.label(t("projects.preflight.checking")).classes("text-sm sam-muted")
                self.checking.set_visibility(False)
            self.headline = ui.label().classes("text-base font-medium").style(
                "font-variant-numeric: tabular-nums").mark("preflight-headline")
            self.detail = ui.label().classes("text-sm sam-muted").style(
                "font-variant-numeric: tabular-nums").mark("preflight-detail")
            self.rows: dict[str, tuple[Any, Any, Any]] = {}
            for name in ("peak", "after", "free"):
                row = ui.row().classes("w-full items-baseline justify-between gap-x-4 text-sm")
                with row:
                    label = ui.label().classes("sam-muted")
                    value = ui.label().style("font-variant-numeric: tabular-nums").mark(f"preflight-{name}")
                self.rows[name] = (row, label, value)
            self.status_row = ui.row().classes("items-center gap-1").mark("preflight-status-row")
            with self.status_row:
                self.status_icon = ui.icon("info").classes("text-lg")
                self.status_label = ui.label().classes("text-sm font-medium").mark("preflight-status")
            self.advice = ui.label().classes("text-sm").mark("preflight-advice")
            self.notes = ui.column().classes("w-full gap-0").mark("preflight-notes")

    # ---------------------------------------------------------------------------------------- scheduling

    def snapshot(self) -> PreflightInputs:
        try:
            interval = float(self._interval.value or 0)
        except (TypeError, ValueError):
            interval = 0.0
        return PreflightInputs(
            capture_type=self._capture_type.value or "",
            image_directory=(self._image_directory.value or "").strip(),
            panorama_video_path=(self._video.value or "").strip(),
            panorama_image_directory=(self._photos.value or "").strip(),
            interval=interval,
            storage=tuple((key, bool(control.value)) for key, control in self._storage.items()),
            projects_directory=str(self._projects_directory()),
        )

    def schedule(self) -> None:
        """Debounced recompute; a newer call invalidates every older run."""
        self.version += 1
        version = self.version
        background_tasks.create(self._run(version, self.snapshot()), name="project-preflight")

    async def _run(self, version: int, inputs: PreflightInputs) -> None:
        await asyncio.sleep(DEBOUNCE_SECONDS)
        if version != self.version or self.root.is_deleted:
            return
        self.checking.set_visibility(True)
        self.root.classes(add="opacity-60")
        try:
            result = await run.io_bound(compute, inputs, self._cache)
        except Exception:  # noqa: BLE001 - a pre-check must never disturb the form
            result = preflight.PreflightResult("unknown", (preflight.Note("unreadable_generic"),), inputs.capture_type)
        if version != self.version or self.root.is_deleted:
            return  # a newer input exists or the client left: never overwrite a newer state
        self.apply(result)

    # --------------------------------------------------------------------------------------- rendering

    def apply(self, result: preflight.PreflightResult) -> None:
        self.checking.set_visibility(False)
        self.root.classes(remove="opacity-60")
        if result.state == "empty":
            self._render_empty()
        elif result.state == "unknown":
            self._render_unknown(result)
        else:
            self._render_ready(result)

    def _hide_numbers(self) -> None:
        for row, _label, _value in self.rows.values():
            row.set_visibility(False)
        self.status_row.set_visibility(False)
        self.advice.set_visibility(False)
        self.detail.set_visibility(False)

    def _render_empty(self) -> None:
        self._hide_numbers()
        self.headline.text = t("projects.preflight.empty")
        self.headline.classes(add="sam-muted", remove="font-medium")
        self._render_notes(())

    def _render_unknown(self, result: preflight.PreflightResult) -> None:
        self._hide_numbers()
        self.headline.text = t("projects.preflight.unknown")
        self.headline.classes(add="sam-muted", remove="font-medium")
        self._render_notes(result.notes)

    def _render_notes(self, notes: tuple[preflight.Note, ...]) -> None:
        self.notes.clear()
        with self.notes:
            for note in notes:
                warning = note.key in _WARNING_NOTES
                with ui.row().classes("items-start gap-1 no-wrap"):
                    ui.icon("warning" if warning else "info").classes(
                        "text-base " + ("text-warning" if warning else "sam-muted"))
                    ui.label(t(f"projects.preflight.note.{note.key}", **dict(note.params))).classes(
                        "text-sm " + ("" if warning else "sam-muted"))

    def _render_ready(self, r: preflight.PreflightResult) -> None:
        lang = current_language()
        count = lambda value: preflight.format_count(value, lang)  # noqa: E731
        size = lambda value: preflight.format_gb(value, lang)  # noqa: E731
        typical = size(r.peak_typical)
        if r.capture_type == "panorama_360":
            self.headline.text = t(
                "projects.preflight.headline_360", panoramas=count(r.panoramas), faces=count(r.faces), size=typical)
        else:
            self.headline.text = t("projects.preflight.headline_images", images=count(r.images), size=typical)
        self.headline.classes(add="font-medium", remove="sam-muted")
        parts: list[str] = []
        if r.video is not None and r.video.status == "ok":
            parts.append(t(
                "projects.preflight.detail_video", duration=preflight.format_duration(r.video.duration),
                width=count(r.video.width), height=count(r.video.height),
                interval=preflight.format_interval(r.interval), count=count(r.video_panoramas)))
        if r.folder is not None and r.folder.status == "ok":
            parts.append(t(
                "projects.preflight.detail_photos" if r.capture_type == "panorama_360"
                else "projects.preflight.detail_folder",
                count=count(r.folder.count), size=size(r.folder.size_bytes)))
        self.detail.text = " · ".join(parts)
        self.detail.set_visibility(bool(parts))

        if r.peak_upper <= r.peak_typical * 1.005:
            peak_text = t("projects.preflight.peak_value", typical=typical)
        else:
            peak_text = t("projects.preflight.peak_range", typical=typical, upper=size(r.peak_upper))
        values = {
            "peak": ("projects.preflight.peak", peak_text),
            "after": ("projects.preflight.after", t("projects.preflight.after_value", size=size(r.after))),
            "free": (
                "projects.preflight.free",
                size(r.free) if r.free is not None else t("projects.preflight.free_unknown"),
            ),
        }
        for name, (row, label, value) in self.rows.items():
            key, text = values[name]
            label.text = t(key, drive=r.free_drive or "?")
            value.text = text
            row.set_visibility(True)
        self._render_status(r, size)
        self._render_notes(r.notes)

    def _render_status(self, r: preflight.PreflightResult, size: Callable[[float], str]) -> None:
        look = {
            "ok": ("check_circle", "text-positive", "projects.preflight.status_ok"),
            "tight": ("warning", "text-warning", "projects.preflight.status_tight"),
            "short": ("error", "text-negative", "projects.preflight.status_short"),
            "unknown": ("info", "sam-muted", "projects.preflight.status_unknown"),
        }
        icon, colour, key = look[r.status]
        self.status_icon.props(f"name={icon}")
        for other in ("text-positive", "text-warning", "text-negative", "sam-muted"):
            self.status_icon.classes(remove=other)
            self.status_label.classes(remove=other)
            self.advice.classes(remove=other)
        self.status_icon.classes(add=colour)
        self.status_label.classes(add=colour)
        self.status_label.text = t(key)
        self.status_row.set_visibility(True)
        lines: list[str] = []
        if r.status == "short":
            lines.append(t("projects.preflight.advice_free", needed=size(r.peak_typical), drive=r.free_drive or "?"))
            if r.advice_interval is not None:
                lines.append(t("projects.preflight.advice_interval",
                               interval=preflight.format_interval(r.advice_interval)))
        elif r.status == "tight":
            lines.append(t("projects.preflight.advice_tight", needed=size(r.peak_upper)))
        self.advice.text = " ".join(lines)
        self.advice.classes(add="sam-muted")
        self.advice.set_visibility(bool(lines))
