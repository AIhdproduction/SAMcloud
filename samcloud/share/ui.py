"""NiceGUI parts of "Punktwolke per QR-Code teilen": button, share dialog, Cloudflare setup dialog, settings card."""

from __future__ import annotations

import base64
import html
import logging
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from nicegui import run, ui

from samcloud.app.i18n import current_language, t, translate_or
from samcloud.core.project_manager import ProjectManager
from samcloud.share import credentials as creds
from samcloud.share import export
from samcloud.share.job import (
    POINTCLOUD,
    PROJECT,
    ShareKind,
    delete_share,
    extend_share,
    get_share,
    share_available,
    share_manager,
    share_state,
)
from samcloud.share.project import project_share_available
from samcloud.share.providers import ProviderError

LOGGER = logging.getLogger(__name__)
FIELD_KEYS = creds.FIELDS


def provider_name(config: dict[str, Any] | None = None) -> str:
    """Provider in effect (environment override, else ``config/share.yaml``)."""
    config = config if config is not None else export.load_share_config()
    return str(os.environ.get("SAMCLOUD_SHARE_PROVIDER") or config.get("provider") or "cloudflare_r2")


def needs_credentials() -> bool:
    """True when the selected provider is Cloudflare R2 and no (complete) credentials are stored."""
    return provider_name() == "cloudflare_r2" and creds.load_credentials() is None


def _format_date(value: Any) -> str:
    try:
        return datetime.fromisoformat(str(value)).astimezone().strftime("%d.%m.%Y")
    except ValueError:
        return str(value)


# ------------------------------------------------------------------------------------------------
# Cloudflare setup
# ------------------------------------------------------------------------------------------------

def open_setup_dialog(on_saved: Any | None = None) -> ui.dialog:
    """Dialog with the five credential fields, step-by-step help, connection test and save."""
    stored = creds.load_credentials() or {}
    fields: dict[str, ui.input] = {}
    with ui.dialog().props("persistent") as dialog, ui.card().classes("w-[40rem] max-w-full gap-2").mark("share-setup-dialog"):
        ui.label(t("share.setup.title")).classes("text-lg font-medium")
        ui.label(t("share.setup.intro")).classes("text-sm")
        with ui.expansion(t("share.guide.title"), icon="help_outline").classes("w-full").mark("share-guide"):
            ui.html(guide_html(), sanitize=False).classes("text-sm")
        for key in FIELD_KEYS:
            field = ui.input(t(f"share.setup.field_{key}"), value=stored.get(key, "")).classes("w-full").mark(f"share-field-{key}")
            if key == "secret_access_key":
                field.props("type=password")
            fields[key] = field
        result = ui.label("").classes("text-sm").mark("share-setup-result")

        def values() -> dict[str, str]:
            return creds.normalize({key: field.value for key, field in fields.items()})

        def check_values() -> dict[str, str] | None:
            data = values()
            if not creds.is_complete(data):
                result.set_text(t("share.setup.incomplete"))
                result.classes(replace="text-sm text-negative")
                return None
            return data

        async def test() -> None:
            data = check_values()
            if data is None:
                return
            result.set_text(t("share.setup.testing"))
            result.classes(replace="text-sm")
            try:
                message, warning = await run.io_bound(_test_connection, data)
            except ProviderError as error:
                result.set_text(str(error))
                result.classes(replace="text-sm text-negative")
                return
            if warning:
                result.set_text(f"{message} {warning}")
                result.classes(replace="text-sm text-warning")
            else:
                result.set_text(message)
                result.classes(replace="text-sm text-positive")

        def save() -> None:
            data = check_values()
            if data is None:
                return
            creds.save_credentials(data)
            dialog.close()
            ui.notify(t("share.setup.saved"), type="positive")
            if on_saved is not None:
                on_saved()

        with ui.row().classes("w-full justify-end gap-2"):
            ui.button(t("share.setup.test"), on_click=test).props("outline").mark("share-test")
            ui.button(t("share.setup.save"), on_click=save).mark("share-save")
            ui.button(t("share.close"), on_click=dialog.close).props("flat").mark("share-setup-close")
    dialog.open()
    return dialog


def _test_connection(data: dict[str, str]) -> tuple[str, str | None]:
    """Run the R2 test; returns ``(message, warning)``. A missing deletion rule is only a warning."""
    from samcloud.share.providers.cloudflare_r2 import CloudflareR2Provider

    days = int(export.load_share_config().get("expiry_days") or 15)
    provider = CloudflareR2Provider(data, expiry_days=days)
    provider.test_connection()
    status = provider.ensure_lifecycle(days)
    if status == "unreadable":  # the token cannot read rules: only an informational note
        return t("share.setup.lifecycle_unknown"), None
    if status == "missing":  # rules were readable and none deletes after <= 15 days
        return t("share.setup.ok_short"), t("share.setup.lifecycle_missing")
    return t("share.setup.ok", days=days), None


_URL = re.compile(r"(https?://[^\s<]+)")


def _inline(text: str) -> str:
    """Escape ``text``; backtick spans become <code>, URLs become links."""
    escaped = html.escape(text, quote=False)
    escaped = re.sub(r"`([^`]+)`", r"<code>\1</code>", escaped)
    return _URL.sub(r'<a href="\1" target="_blank" rel="noopener noreferrer">\1</a>', escaped)


def guide_html() -> str:
    """The numbered Cloudflare R2 guide in the current UI language."""
    parts = [f"<h4><b>{_inline(t('share.guide.title'))}</b></h4>"]
    for number in range(1, 7):
        parts.append(f"<p><b>{number}. {_inline(t(f'share.guide.s{number}.title'))}</b></p><ol>")
        parts.extend(f"<li>{_inline(line)}</li>" for line in t(f"share.guide.s{number}.body").split("\n") if line)
        parts.append("</ol>")
    parts.append(f"<p><b>{_inline(t('share.guide.note'))}</b></p>")
    return "".join(parts)


def render_settings_card() -> None:
    """Small "Teilen / Cloudflare" section of the settings page: change or delete the R2 account."""
    with ui.card().classes("w-full").mark("share-settings-card"):
        ui.label(t("share.settings.title")).classes("text-lg font-medium")
        status = ui.label("").classes("text-sm").mark("share-settings-status")

        def refresh() -> None:
            stored = creds.load_credentials()
            if stored:
                status.set_text(t("share.settings.configured", bucket=stored["bucket"], account=stored["account_id"]))
            else:
                status.set_text(t("share.settings.not_configured"))
            delete_button.set_enabled(stored is not None)

        def remove() -> None:
            creds.delete_credentials()
            ui.notify(t("share.settings.deleted"))
            refresh()

        with ui.row().classes("gap-2"):
            ui.button(t("share.settings.edit"), on_click=lambda: open_setup_dialog(refresh)).props("outline").mark("share-settings-edit")
            delete_button = ui.button(t("share.settings.delete"), on_click=remove).props("outline color=negative").mark("share-settings-delete")
        refresh()


# ------------------------------------------------------------------------------------------------
# share dialog
# ------------------------------------------------------------------------------------------------

class ShareDialog:
    """Progress, result (QR + link) and management of the share link of one project."""

    def __init__(self, project_dir: Path, kind: ShareKind = POINTCLOUD) -> None:
        self.project_dir = Path(project_dir)
        self.kind = kind
        self.dialog: ui.dialog | None = None
        self.body: ui.column | None = None
        self.timer: ui.timer | None = None
        self._view: tuple[Any, ...] | None = None
        self._bar: ui.linear_progress | None = None
        self._text: ui.label | None = None

    # opening -----------------------------------------------------------------------------------
    def open(self) -> None:
        with ui.dialog().props("persistent") as self.dialog, ui.card().classes("w-[34rem] max-w-full gap-3").mark("share-dialog"):
            ui.label(self._t("title")).classes("text-lg font-medium")
            self.body = ui.column().classes("w-full gap-2")
        self.dialog.on("hide", self._stop_timer)
        self.timer = ui.timer(1.0, self.poll)
        self.dialog.open()
        self.render()

    def close(self) -> None:
        self._stop_timer()
        if self.dialog is not None:
            self.dialog.close()

    def _stop_timer(self, *_: Any) -> None:
        if self.timer is not None:
            self.timer.cancel()

    def _t(self, key: str, **params: Any) -> str:
        """Translate a share key in the namespace for this kind, with point-cloud fallback."""
        fallback = t(f"share.{key}", **params)
        if self.kind is POINTCLOUD:
            return fallback
        return translate_or(f"share.project.{key}", fallback, **params)

    # state -------------------------------------------------------------------------------------
    def _state_key(self) -> tuple[Any, ...]:
        progress = share_manager.status(self.project_dir, kind=self.kind)
        block = get_share(self.project_dir, kind=self.kind)
        return (progress.get("state"), share_state(block), (block or {}).get("share_id"), progress.get("error"),
                progress.get("phase_index"), tuple(progress.get("phases_done") or ()))

    def _phase_name(self, progress: dict[str, Any]) -> tuple[str, str]:
        """Return (translated phase name, German phase label) of the active phase."""
        label = str(progress.get("phase") or "")
        try:
            index = int(progress.get("phase_index", -1))
        except (TypeError, ValueError):
            index = -1
        if 0 <= index < len(self.kind.phases):
            return self._t(f"phase_{self.kind.phases[index][0]}"), str(self.kind.phases[index][1])
        return label, label

    def _progress_text(self, progress: dict[str, Any]) -> str:
        name, label = self._phase_name(progress)
        message = str(progress.get("message") or "").strip()
        if label and message.startswith(label):
            message = message[len(label):].strip()
        if message in ("", label, name):
            message = ""
        elif current_language() != "de" and re.search(r"[^\W\d_]", re.sub(r"\b[kKMG]?B\b", "", message)):
            message = ""  # German free text is not shown in other UI languages
        percent = f"({float(progress.get('percent', 0)):.0f} %)"
        head = f"{name}: {message}" if message else name
        return f"{head} {percent}".strip()

    def poll(self) -> None:
        if self._state_key() != self._view:
            self.render()
            return
        progress = share_manager.status(self.project_dir, kind=self.kind)
        if progress.get("state") == "running" and self._bar is not None and self._text is not None:
            self._bar.set_value(float(progress.get("percent", 0)) / 100.0)
            self._text.set_text(self._progress_text(progress))

    def render(self) -> None:
        assert self.body is not None
        self.body.clear()
        self._bar = self._text = None
        self._view = self._state_key()
        progress = share_manager.status(self.project_dir, kind=self.kind)
        block = get_share(self.project_dir, kind=self.kind)
        with self.body:
            if progress.get("state") == "running":
                self._render_running(progress)
            elif block is not None:
                self._render_result(block, share_state(block))
            elif progress.get("state") == "error":
                self._render_message(str(progress.get("error") or self._t("error_unknown")), error=True)
            elif progress.get("state") == "cancelled":
                self._render_message(self._t("cancelled"))
            else:
                self._render_start()

    # views -------------------------------------------------------------------------------------
    def _footer(self, *buttons: tuple[str, Any, str]) -> None:
        with ui.row().classes("w-full justify-end gap-2"):
            for label, handler, marker in buttons:
                ui.button(label, on_click=handler).props("outline" if marker != "share-start" else "").mark(marker)
            ui.button(self._t("close"), on_click=self.close).props("flat").mark("share-close")

    def _render_start(self) -> None:
        max_points = f"{int(export.load_share_config()['max_points']):,}".replace(",", "'")
        ui.label(self._t("intro", max_points=max_points)).classes("text-sm")
        self._footer((self._t("start"), self.start, "share-start"))

    def _render_message(self, message: str, *, error: bool = False) -> None:
        ui.label(message).classes("text-sm text-negative" if error else "text-sm").mark("share-message")
        self._footer((self._t("retry"), self.start, "share-start"))

    def _render_running(self, progress: dict[str, Any]) -> None:
        done = {str(key) for key in progress.get("phases_done") or []}  # phase keys; the active phase is included
        active = progress.get("phase_index", -1)
        for index, label in enumerate(progress.get("phases", [])):
            phase_key = self.kind.phases[index][0] if index < len(self.kind.phases) else ""
            text = self._t(f"phase_{phase_key}") if phase_key else str(label)
            if index == active:
                ui.label("• " + text).classes("text-sm font-bold")
            elif phase_key in done:
                ui.label("✓ " + text).classes("text-sm")
            else:
                ui.label("◦ " + text).classes("text-sm sam-muted")
        self._bar = ui.linear_progress(value=float(progress.get("percent", 0)) / 100.0, show_value=False).mark("share-progress")
        self._text = ui.label(self._progress_text(progress)).classes("text-sm").mark("share-progress-text")
        with ui.row().classes("w-full justify-end"):
            ui.button(self._t("cancel"), on_click=self.cancel).props("outline color=negative").mark("share-cancel")

    def _render_result(self, block: dict[str, Any], state: str) -> None:
        expired = state == "expired"
        if expired:
            ui.label(self._t("expired", date=_format_date(block.get("expires_at")))).classes("text-negative").mark("share-expired")
        else:
            qr = self.project_dir / self.kind.output_relative / "qr.png"
            if qr.is_file():
                encoded = base64.b64encode(qr.read_bytes()).decode("ascii")
                ui.image(f"data:image/png;base64,{encoded}").classes("w-64 h-64 self-center bg-white").mark("share-qr")
            ui.link(str(block["url"]), str(block["url"]), new_tab=True).classes("break-all text-sm").mark("share-link")
            if block.get("expires_at"):
                ui.label(self._t("valid_until", date=_format_date(block["expires_at"]))).classes("text-sm").mark("share-valid-until")
            ui.label(self._t("info", points=f"{int(block.get('points', 0)):,}".replace(",", "'"), size=block.get("size_mb", 0))).classes("text-xs sam-muted")
            if self.kind is PROJECT:
                labels = {"dxf": self._t("view_dxf"), "pc": self._t("view_pc"), "ifc": self._t("view_ifc")}
                views = [labels[view] for view in block.get("views", []) if view in labels]
                if views:
                    details = self._t("views_line", views=" · ".join(views))
                    levels = block.get("levels", [])
                    if isinstance(levels, list) and len(levels) > 1:
                        details += f" · {self._t('levels', count=len(levels))}"
                    ui.label(details).classes("text-xs sam-muted").mark("share-project-views")
            with ui.row().classes("gap-2"):
                ui.button(self._t("copy"), on_click=lambda: self._copy(str(block["url"]))).props("outline dense").mark("share-copy")
                ui.button(self._t("save_png"), on_click=self._save_png).props("outline dense").mark("share-save-png")
                ui.button(self._t("open"), on_click=lambda: ui.navigate.to(str(block["url"]), new_tab=True)).props("outline dense").mark("share-open")
        buttons: list[tuple[str, Any, str]] = []
        if not expired and block.get("expires_at"):
            buttons.append((self._t("extend"), self.extend, "share-extend"))
        buttons.append((self._t("recreate"), self.recreate, "share-recreate"))
        buttons.append((self._t("delete"), self.delete, "share-delete"))
        self._footer(*buttons)

    # actions -----------------------------------------------------------------------------------
    def start(self) -> None:
        """Ask for the privacy confirmation once per project, then start the job."""
        manager = ProjectManager()
        project = manager.load(self.project_dir)
        confirmation_key = "share_confirmed" if self.kind is POINTCLOUD else "project_share_confirmed"
        if project.get("settings", {}).get(confirmation_key):
            self._launch()
            return
        with ui.dialog() as confirm, ui.card().mark("share-confirm"):
            ui.label(self._t("confirm")).classes("text-sm")
            with ui.row().classes("justify-end w-full gap-2"):
                ui.button(self._t("confirm_no"), on_click=confirm.close).props("flat").mark("share-confirm-no")
                ui.button(self._t("confirm_yes"), on_click=lambda: self._confirmed(confirm)).mark("share-confirm-yes")
        confirm.open()

    def _confirmed(self, confirm: ui.dialog) -> None:
        manager = ProjectManager()
        project = manager.load(self.project_dir)
        confirmation_key = "share_confirmed" if self.kind is POINTCLOUD else "project_share_confirmed"
        project.setdefault("settings", {})[confirmation_key] = True
        manager.save(self.project_dir, project)
        confirm.close()
        self._launch()

    def _launch(self) -> None:
        try:
            share_manager.start(self.project_dir, kind=self.kind)
        except Exception as error:  # e.g. missing credentials: show it, never hang
            LOGGER.warning("share start failed: %s", error)
            ui.notify(str(error), type="negative")
        self.render()

    def cancel(self) -> None:
        share_manager.cancel(self.project_dir, kind=self.kind)
        ui.notify(self._t("cancelling"))

    async def extend(self) -> None:
        try:
            block = await run.io_bound(extend_share, self.project_dir, kind=self.kind)
        except ProviderError as error:
            ui.notify(str(error), type="negative")
            return
        ui.notify(self._t("extended", date=_format_date(block.get("expires_at"))), type="positive")
        self.render()

    def recreate(self) -> None:
        self._launch()

    async def delete(self) -> None:
        try:
            await run.io_bound(delete_share, self.project_dir, kind=self.kind)
        except ProviderError as error:
            ui.notify(str(error), type="negative")
            return
        ui.notify(self._t("deleted"))
        self.render()

    def _copy(self, url: str) -> None:
        ui.run_javascript(f"navigator.clipboard.writeText({url!r})")
        ui.notify(self._t("copied"))

    def _save_png(self) -> None:
        qr = self.project_dir / self.kind.output_relative / "qr.png"
        if qr.is_file():
            ui.download(qr.read_bytes(), "samcloud-share-qr.png")


# ------------------------------------------------------------------------------------------------
# button
# ------------------------------------------------------------------------------------------------

def render_share_button(project_dir: Path | None, project: dict[str, Any] | None) -> ui.button | None:
    """"Teilen (QR)" button: disabled with a tooltip until step 6 is done; opens setup or share dialog."""
    if project_dir is None:
        return None
    available = share_available(project_dir, project)
    button = ui.button(t("share.button"), icon="qr_code_2").props("outline dense no-caps").mark("share-button")
    if not available:
        button.props("disable")
        with button:
            ui.tooltip(t("share.unavailable"))
        return button

    def open_share() -> None:
        ShareDialog(project_dir).open()

    def clicked() -> None:
        if needs_credentials():
            open_setup_dialog(open_share)
        else:
            open_share()

    button.on_click(clicked)
    return button


def render_project_share_button(project_dir: Path | None, project: dict[str, Any] | None) -> ui.button | None:
    """Render the complete-project QR sharing button when all project views are available."""
    if project_dir is None or not project_share_available(project_dir, project):
        return None
    button = ui.button(t("share.project.button"), icon="qr_code_2").props("outline dense no-caps").classes("w-full q-mt-sm").mark("project-share-button")

    def open_share() -> None:
        ShareDialog(project_dir, kind=PROJECT).open()

    def clicked() -> None:
        if needs_credentials():
            open_setup_dialog(open_share)
        else:
            open_share()

    button.on_click(clicked)
    return button
