"""Settings card of the optional AI assistant: provider, server address / API key, model, cost limits, full control,
connection test.

Full control (``full_control``): the switch shows the effective value; while the user has not chosen, it follows the
provider default (on for Claude / OpenAI / OpenRouter, off for Ollama) and nothing explicit is stored; a click stores the
explicit choice, which a later provider change never overwrites ("Use the default" goes back to the provider default).

Nothing is written when the page is only opened; "Save" writes ``ai_settings.yaml`` and ``ai_credentials.yaml`` in the
user config directory (never a repository file). Stored keys are only ever shown masked.
"""

from __future__ import annotations

import logging
from dataclasses import replace
from typing import Any

from nicegui import run, ui

from samcloud.app.i18n import t
from samcloud.assistant import credentials as ai_creds
from samcloud.assistant.costs import describe_price, price_for
from samcloud.assistant.providers import ConnectionResult, ModelInfo, ProviderError, get_provider
from samcloud.assistant.providers.ollama import model_warnings, rank_models, recommended_model
from samcloud.assistant.settings import (
    CLOUD_PROVIDERS,
    OLLAMA_NUM_CTX_AUTO,
    PROVIDERS,
    AssistantSettings,
    effective_full_control,
    is_active,
    load_settings,
    normalize_num_ctx,
    save_settings,
)

LOGGER = logging.getLogger(__name__)

NUM_CTX_CHOICES: tuple[int, ...] = (8192, 16384, 32768, 65536, 131072)


def provider_options() -> dict[str, str]:
    """Select options in UI order (Ollama first)."""
    return {name: t(f"assistant.settings.provider.{name}") for name in PROVIDERS}


def num_ctx_options(current: str | None = None) -> dict[str, str]:
    """Options of the Ollama context-window select (``"auto"`` first; a stored custom value is kept)."""
    options = {OLLAMA_NUM_CTX_AUTO: t("assistant.settings.num_ctx_auto")}
    values = list(NUM_CTX_CHOICES)
    custom = normalize_num_ctx(current)
    if custom != OLLAMA_NUM_CTX_AUTO and int(custom) not in values:
        values = sorted([*values, int(custom)])
    for value in values:
        options[str(value)] = t("assistant.settings.num_ctx_value", tokens=f"{value:,}".replace(",", "\u2009"))
    return options


def model_label(info: ModelInfo, recommended: str | None = None) -> str:
    """Select label of a model: id, context window, size, cloud / recommended marks."""
    parts = [info.id]
    if info.context_length:
        parts.append(t("assistant.settings.model_context", k=int(round(info.context_length / 1024))))
    if info.size_bytes and not info.remote:
        parts.append(f"{info.size_bytes / 1e9:.1f} GB")
    if info.remote:
        parts.append(t("assistant.settings.model_remote"))
    if recommended and info.id == recommended:
        parts.append(t("assistant.settings.model_recommended"))
    return " \u00b7 ".join(parts)


def warning_text(info: ModelInfo | None, provider_name: str) -> str:
    """Warnings for the chosen model (no / unknown tool support, small context window, Ollama cloud model)."""
    codes = model_warnings(info)
    if provider_name != "ollama":
        codes = [code for code in codes if code in ("no_tools", "small_context")]
    return " ".join(t(f"assistant.settings.model_warning.{code}") for code in codes)


def _number(value: Any) -> float:
    try:
        return max(0.0, float(value))
    except (TypeError, ValueError):
        return 0.0


def render_assistant_settings_card() -> None:
    """The "KI-Assistent / AI assistant" card of the settings page."""
    saved = load_settings()
    stored = ai_creds.load_ai_credentials()
    infos: dict[str, ModelInfo] = {}
    form: dict[str, Any] = {"settings": saved, "creds": stored}

    with ui.card().classes("w-full gap-2").mark("assistant-settings-card"):
        ui.label(t("assistant.settings.title")).classes("text-lg font-medium")
        ui.label(t("assistant.settings.intro")).classes("text-sm sam-muted")
        enabled = ui.switch(t("assistant.settings.enabled"), value=saved.enabled).mark("assistant-enabled")
        provider = ui.select(provider_options(), label=t("assistant.settings.provider_label"),
                             value=saved.provider).classes("min-w-[16rem]").mark("assistant-provider")

        with ui.column().classes("w-full gap-1").mark("assistant-ollama-box") as ollama_box:
            ollama_url = ui.input(t("assistant.settings.ollama_url"), value=stored["ollama_url"]).classes(
                "w-full max-w-[32rem]").mark("assistant-ollama-url")
            ui.label(t("assistant.settings.ollama_hint")).classes("text-sm sam-muted").mark("assistant-ollama-hint")
            ollama_ctx = ui.select(num_ctx_options(saved.ollama_num_ctx), label=t("assistant.settings.num_ctx"),
                                   value=normalize_num_ctx(saved.ollama_num_ctx)).classes("min-w-[16rem]").mark(
                "assistant-ollama-num-ctx")
            ui.label(t("assistant.settings.num_ctx_hint")).classes("text-sm sam-muted").mark("assistant-num-ctx-hint")

        with ui.column().classes("w-full gap-1").mark("assistant-cloud-box") as cloud_box:
            key_status = ui.label("").classes("text-sm").mark("assistant-key-status")
            with ui.row().classes("items-center gap-2 w-full"):
                api_key = ui.input(t("assistant.settings.api_key"), placeholder=t("assistant.settings.api_key_hint")).props(
                    "type=password autocomplete=off").classes("min-w-[20rem]").mark("assistant-api-key")
                delete_key = ui.button(t("assistant.settings.delete_key")).props(
                    "outline color=negative no-caps").mark("assistant-delete-key")
            with ui.row().classes("gap-2"):
                cost_request = ui.number(t("assistant.settings.cost_request"), value=saved.cost_limit_request_usd,
                                         min=0, step=0.1, format="%.2f").classes("w-64").mark("assistant-cost-request")
                cost_day = ui.number(t("assistant.settings.cost_day"), value=saved.cost_limit_project_day_usd,
                                     min=0, step=0.5, format="%.2f").classes("w-64").mark("assistant-cost-day")
            ui.label(t("assistant.settings.cost_hint")).classes("text-sm sam-muted")

        with ui.row().classes("items-center gap-2 w-full"):
            initial_options = {saved.model: saved.model} if saved.model else {}
            model_select = ui.select(initial_options, label=t("assistant.settings.model_select"),
                                     value=saved.model or None, with_input=True).classes(
                "min-w-[20rem]").mark("assistant-model-select")
            load_button = ui.button(t("assistant.settings.load_models")).props("outline no-caps").mark(
                "assistant-load-models")
        model_text = ui.input(t("assistant.settings.model_text"), value="").classes(
            "w-full max-w-[32rem]").mark("assistant-model")
        models_status = ui.label("").classes("text-sm").mark("assistant-models-status")
        model_warning = ui.label("").classes("text-sm text-warning").mark("assistant-model-warning")
        price_label = ui.label("").classes("text-sm sam-muted").mark("assistant-price")
        text_only = ui.switch(t("assistant.settings.text_only"), value=saved.text_only).mark("assistant-text-only")
        text_only_hint = ui.label(t("assistant.settings.text_only_forced")).classes("text-sm sam-muted").mark(
            "assistant-text-only-hint")
        full_state: dict[str, Any] = {"explicit": saved.full_control, "syncing": False}
        with ui.column().classes("w-full gap-1").mark("assistant-full-control-box"):
            full_control = ui.switch(t("assistant.settings.full_control"), value=effective_full_control(saved)).mark(
                "assistant-full-control")
            ui.label(t("assistant.settings.full_control_hint")).classes("text-sm sam-muted").mark("assistant-full-control-hint")
            with ui.row().classes("items-center gap-2"):
                full_default = ui.label("").classes("text-sm sam-muted").mark("assistant-full-control-default")
                full_reset = ui.button(t("assistant.settings.full_control_reset")).props("flat dense no-caps").mark(
                    "assistant-full-control-reset")
        with ui.row().classes("gap-2"):
            test_button = ui.button(t("assistant.settings.test_button")).props("outline no-caps").mark("assistant-test")
            save_button = ui.button(t("assistant.settings.save")).props("no-caps").mark("assistant-save")
        test_result = ui.label("").classes("text-sm").mark("assistant-test-result")
        save_status = ui.label("").classes("text-sm").mark("assistant-save-status")

        # ------------------------------------------------------------------ helpers

        def current_provider() -> str:
            return str(provider.value or "ollama")

        def effective_model() -> str:
            return str(model_text.value or "").strip() or str(model_select.value or "").strip()

        def form_settings() -> AssistantSettings:
            base: AssistantSettings = form["settings"]
            return replace(base, enabled=bool(enabled.value), provider=current_provider(), model=effective_model(),
                           text_only=bool(text_only.value), cost_limit_request_usd=_number(cost_request.value),
                           cost_limit_project_day_usd=_number(cost_day.value),
                           ollama_num_ctx=normalize_num_ctx(ollama_ctx.value), full_control=full_state["explicit"])

        def form_creds() -> dict[str, str]:
            data = dict(form["creds"])
            data["ollama_url"] = str(ollama_url.value or "").strip().rstrip("/") or ai_creds.DEFAULT_OLLAMA_URL
            name = current_provider()
            typed = str(api_key.value or "").strip()
            if name in CLOUD_PROVIDERS and typed:
                data[ai_creds.KEY_FIELDS[name]] = typed
            return data

        def refresh_key_status() -> None:
            name = current_provider()
            if name not in CLOUD_PROVIDERS:
                return
            masked = ai_creds.masked_ai_credentials()[ai_creds.KEY_FIELDS[name]]
            key_status.set_text(t("assistant.settings.key_stored", masked=masked) if masked
                                else t("assistant.settings.key_missing"))
            delete_key.set_enabled(bool(masked))

        def refresh_model_capabilities(*_: Any) -> None:
            info = infos.get(effective_model())
            forced = info is not None and info.supports_images is False
            if forced:
                text_only.value = True
            text_only.set_enabled(not forced)
            text_only_hint.set_visibility(forced)
            warning = warning_text(info, current_provider()) if effective_model() else ""
            model_warning.set_text(warning)
            model_warning.set_visibility(bool(warning))
            if current_provider() in CLOUD_PROVIDERS and effective_model():
                price_label.set_text(describe_price(price_for(info or effective_model(), form_settings())))
            else:
                price_label.set_text("")

        def refresh_full_control() -> None:
            name = current_provider()
            default = name in CLOUD_PROVIDERS
            explicit = full_state["explicit"]
            state = t("assistant.settings.full_control_on" if default else "assistant.settings.full_control_off")
            text = t("assistant.settings.full_control_default",
                     provider=t(f"assistant.settings.provider.{name}"), state=state)
            if explicit is not None:
                text = f"{text} {t('assistant.settings.full_control_explicit')}"
            full_default.set_text(text)
            full_reset.set_visibility(explicit is not None)
            effective = default if explicit is None else bool(explicit)
            if bool(full_control.value) != effective:
                full_state["syncing"] = True
                try:
                    full_control.value = effective
                finally:
                    full_state["syncing"] = False

        def full_control_changed(*_: Any) -> None:
            if full_state["syncing"]:
                return
            full_state["explicit"] = bool(full_control.value)
            refresh_full_control()

        def reset_full_control() -> None:
            full_state["explicit"] = None
            refresh_full_control()

        def refresh_provider(*_: Any) -> None:
            name = current_provider()
            ollama_box.set_visibility(name == "ollama")
            cloud_box.set_visibility(name in CLOUD_PROVIDERS)
            refresh_key_status()
            refresh_model_capabilities()
            refresh_full_control()

        def provider_changed(*_: Any) -> None:
            infos.clear()
            restore = saved.model if current_provider() == saved.provider and saved.model else None
            model_select.set_options({restore: restore} if restore else {}, value=restore)
            api_key.value = ""
            models_status.set_text("")
            test_result.set_text("")
            refresh_provider()

        def show(label: ui.label, text: str, tone: str | None = None) -> None:
            label.set_text(text)
            label.classes(replace="text-sm" + (f" text-{tone}" if tone else ""))

        def build_provider() -> Any:
            return get_provider(current_provider(), form_settings(), form_creds())

        # ------------------------------------------------------------------ actions

        async def load_models() -> None:
            show(models_status, t("assistant.settings.loading_models"))
            try:
                adapter = build_provider()
                models = await run.io_bound(adapter.list_models)
            except ProviderError as error:
                show(models_status, str(error), "negative")
                return
            except Exception as error:  # never crash the page
                show(models_status, t("assistant.settings.error.unexpected", detail=ai_creds.redact(error)), "negative")
                return
            recommended = None
            hidden = 0
            if current_provider() == "ollama":
                ranked = rank_models(models)  # tool-capable models only, most suitable first
                hidden = len(models) - len(ranked)
                models = ranked
                recommended = recommended_model(models)
            infos.clear()
            infos.update({info.id: info for info in models})
            keep = model_select.value if model_select.value in infos else None
            if keep is None and recommended and not effective_model():
                keep = recommended  # nothing chosen yet: preselect the recommendation
            model_select.set_options({info.id: model_label(info, recommended) for info in models}, value=keep)
            if models:
                text = t("assistant.settings.models_loaded", count=len(models))
                if hidden:
                    text = f"{text} {t('assistant.settings.models_hidden', count=hidden)}"
                if recommended:
                    text = f"{text} {t('assistant.settings.models_recommended', model=recommended)}"
                show(models_status, text, "positive")
            elif hidden:
                show(models_status, t("assistant.settings.models_no_tools", count=hidden), "warning")
            else:
                show(models_status, t("assistant.settings.models_none"), "warning")
            refresh_model_capabilities()

        async def test_connection() -> None:
            show(test_result, t("assistant.settings.testing"))
            try:
                adapter = build_provider()
                result: ConnectionResult = await run.io_bound(adapter.test_connection, effective_model() or None)
            except ProviderError as error:
                result = ConnectionResult(False, error.kind, str(error))
            except Exception as error:  # never crash the page
                result = ConnectionResult(False, "error", t("assistant.settings.error.unexpected",
                                                            detail=ai_creds.redact(error)))
            if result.model_info is not None:
                infos[result.model_info.id] = result.model_info
                refresh_model_capabilities()
            headline = t(f"assistant.settings.test.kind.{result.kind}")
            show(test_result, f"{headline}: {ai_creds.redact(result.message)}", "positive" if result.ok else "negative")

        def remove_key() -> None:
            name = current_provider()
            if name in CLOUD_PROVIDERS and ai_creds.delete_ai_key(name):
                form["creds"] = ai_creds.load_ai_credentials()
                ui.notify(t("assistant.settings.key_deleted"))
            refresh_key_status()

        def write() -> None:
            settings = form_settings()
            name = settings.provider
            typed = str(api_key.value or "").strip()
            stored_now = ai_creds.load_ai_credentials()
            url = str(ollama_url.value or "").strip().rstrip("/") or ai_creds.DEFAULT_OLLAMA_URL
            changes: dict[str, str] = {}
            if url != stored_now["ollama_url"]:
                changes["ollama_url"] = url
            if name in CLOUD_PROVIDERS and typed and typed != stored_now[ai_creds.KEY_FIELDS[name]]:
                changes[ai_creds.KEY_FIELDS[name]] = typed
            if changes:
                ai_creds.save_ai_credentials(changes)
            save_settings(settings)
            form["settings"] = settings
            form["creds"] = ai_creds.load_ai_credentials()
            api_key.value = ""
            refresh_key_status()
            active = is_active(settings, form["creds"])
            message = t("assistant.settings.saved") if active or not settings.enabled else t(
                "assistant.settings.saved_inactive")
            show(save_status, message, "positive" if active or not settings.enabled else "warning")
            ui.notify(message, type="positive" if active or not settings.enabled else "warning")

        def save() -> None:
            settings = form_settings()
            if settings.enabled and settings.provider in CLOUD_PROVIDERS and not form["settings"].privacy_ack_cloud:
                open_privacy_dialog()
                return
            write()

        def open_privacy_dialog() -> None:
            with ui.dialog().props("persistent") as dialog, ui.card().classes("w-[34rem] max-w-full gap-3").mark(
                    "assistant-privacy-dialog"):
                ui.label(t("assistant.settings.privacy.title")).classes("text-lg font-medium")
                ui.label(t("assistant.settings.privacy.body")).classes("text-sm")

                def confirm() -> None:
                    form["settings"] = replace(form["settings"], privacy_ack_cloud=True)
                    dialog.close()
                    write()

                def cancel() -> None:
                    dialog.close()
                    show(save_status, t("assistant.settings.not_saved"), "warning")

                with ui.row().classes("w-full justify-end gap-2"):
                    ui.button(t("assistant.settings.privacy.cancel"), on_click=cancel).props("flat").mark(
                        "assistant-privacy-cancel")
                    ui.button(t("assistant.settings.privacy.confirm"), on_click=confirm).mark("assistant-privacy-confirm")
            dialog.open()

        test_button.on_click(test_connection)
        save_button.on_click(save)
        provider.on_value_change(provider_changed)
        model_select.on_value_change(refresh_model_capabilities)
        model_text.on_value_change(refresh_model_capabilities)
        load_button.on_click(load_models)
        delete_key.on_click(remove_key)
        full_control.on_value_change(full_control_changed)
        full_reset.on_click(reset_full_control)
        refresh_provider()
