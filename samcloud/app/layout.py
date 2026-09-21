"""Shared page frame and navigation."""

from collections.abc import Callable

from nicegui import ui


_NAVIGATION = (
    ("Overview", "/", "home"),
    ("Projects", "/projects", "folder"),
    ("Sparse cloud", "/sparse", "hub"),
    ("Control points", "/control-points", "my_location"),
    ("Dense cloud", "/dense", "view_in_ar"),
    ("Settings", "/settings", "tune"),
)


_STYLE = """
<style>
:root { --sam-bg:#09131d; --sam-panel:#0e1c28; --sam-line:#20394b; --sam-line-soft:rgba(113,154,181,.18); --sam-text:#eaf2f8; --sam-muted:#96aabe; --sam-accent:#26b7ef; --sam-success:#4bc692; }
body { background:var(--sam-bg)!important; color:var(--sam-text); }
.sam-shell { min-height:100vh; width:100%; background:var(--sam-bg); color:var(--sam-text); }
.sam-sidebar { width:260px; min-width:260px; min-height:100vh; padding:30px 16px 22px; border-right:1px solid var(--sam-line); background:linear-gradient(180deg,#0a1824 0%,#0a1520 100%); }
.sam-wordmark { margin:0 14px 38px; font-size:32px; line-height:1; letter-spacing:-1.4px; color:var(--sam-text); }.sam-wordmark strong { color:var(--sam-accent); font-weight:800; }
.sam-nav { width:100%; min-height:52px; padding:0 14px; margin:3px 0; justify-content:flex-start; color:#d5e1eb!important; border-radius:8px; font-size:15px; }.sam-nav .q-icon { margin-right:13px; font-size:21px; color:#a8bfd1; }.sam-nav:hover { background:rgba(55,137,182,.14); }
.sam-nav-active { color:#effbff!important; background:rgba(36,150,207,.22); border-left:3px solid var(--sam-accent); border-radius:0 8px 8px 0; padding-left:11px; }.sam-nav-active .q-icon { color:var(--sam-accent); }
.sam-sidebar-footer { margin:auto 11px 0; padding-top:24px; color:#7890a5; font-size:10px; letter-spacing:1.5px; }
.sam-main { min-width:0; flex:1 1 auto; }.sam-topbar { height:32px; border-bottom:1px solid var(--sam-line-soft); }.sam-page { width:min(1220px,100%); margin:0 auto; padding:46px clamp(22px,4vw,58px) 64px; }
.sam-page-title { margin:0 0 31px; font-size:clamp(30px,3vw,42px); font-weight:650; letter-spacing:-.8px; line-height:1.12; color:var(--sam-text); }.sam-section-title { color:var(--sam-text); font-size:20px; font-weight:600; letter-spacing:-.2px; }.sam-muted { color:var(--sam-muted); }
.sam-surface, .q-card { background:var(--sam-panel)!important; border:1px solid var(--sam-line)!important; border-radius:9px!important; box-shadow:none!important; color:var(--sam-text); }
.sam-dropzone { min-height:390px; padding:42px 30px; border:1px dashed #47718a; border-radius:9px; background:radial-gradient(circle at 50% 25%,rgba(24,93,126,.14),transparent 45%),#0c1924; }.sam-dropzone .q-icon { color:#c2d7e6; font-size:64px; }.sam-dropzone-title { margin-top:18px; font-size:21px; font-weight:520; color:var(--sam-text); }
.sam-button-primary { background:var(--sam-accent)!important; color:#06202f!important; min-height:46px; padding:0 18px; font-weight:700; border-radius:7px; }.sam-button-secondary { color:#8cdbfa!important; border-color:var(--sam-accent)!important; min-height:46px; padding:0 18px; border-radius:7px; }
.sam-workflow { padding:25px 22px; }.sam-workflow-title { margin-bottom:16px; color:var(--sam-text); font-size:19px; font-weight:600; }.sam-workflow-step { position:relative; min-height:66px; padding:10px 0 10px 50px; color:#c9d8e4; }.sam-workflow-step:not(:last-child)::before { content:''; position:absolute; left:17px; top:43px; height:28px; border-left:1px dashed #537189; }
.sam-step-number { position:absolute; left:0; top:7px; width:35px; height:35px; display:grid; place-items:center; border:1px solid #5f7a90; border-radius:50%; color:#d9e9f4; font-size:13px; }.sam-workflow-step.is-active { margin:0 -8px; padding-left:58px; border:1px solid #24719b; border-radius:7px; background:rgba(24,105,150,.14); }.sam-workflow-step.is-active .sam-step-number { left:8px; border-color:var(--sam-accent); background:var(--sam-accent); color:#06202f; }.sam-workflow-step.is-complete .sam-step-number { border-color:var(--sam-success); color:var(--sam-success); }
.sam-step-label { display:block; font-size:15px; }.sam-step-status { display:block; margin-top:3px; color:var(--sam-muted); font-size:12px; text-transform:capitalize; }
.sam-recent { margin-top:22px; padding:20px 22px 10px; }.sam-recent-head { padding-bottom:16px; border-bottom:1px solid var(--sam-line); }.sam-project-row { padding:16px 2px; border-bottom:1px solid var(--sam-line-soft); }.sam-project-row:last-child { border-bottom:0; }.sam-project-name { color:var(--sam-text); font-weight:600; }.sam-project-path { color:var(--sam-muted); font-size:13px; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }.sam-empty-row { padding:22px 2px; color:var(--sam-muted); }
.q-field__label,.q-field__native,.q-field__input,.q-select__dropdown-icon { color:var(--sam-text)!important; }.q-field--outlined .q-field__control:before { border-color:#456176!important; }.q-field--outlined .q-field__control:hover:before { border-color:var(--sam-accent)!important; }.q-toggle__label,.q-checkbox__label,.q-radio__label { color:var(--sam-text); }.q-btn { text-transform:none; letter-spacing:0; }
@media (max-width:820px) { .sam-sidebar { width:58px; min-width:58px; padding:24px 5px; }.sam-wordmark { margin:0 0 32px; font-size:0; text-align:center; }.sam-wordmark strong { font-size:26px; }.sam-nav { min-width:48px; padding:0; justify-content:center; font-size:0; }.sam-nav .q-btn__content { font-size:0; }.sam-nav .q-icon { margin:0; font-size:21px!important; }.sam-nav-active { padding-left:0; border-left-width:2px; }.sam-sidebar-footer { display:none; }.sam-page { padding:30px 18px 48px; }.sam-page-title { font-size:29px; }.sam-dropzone { min-height:320px; } }
</style>
"""


def render_page(title: str, content: Callable[[], None], *, current_path: str) -> None:
    """Render the common navigation shell around a page's functional content."""
    ui.add_head_html(_STYLE)
    ui.colors(primary="#26b7ef", positive="#4bc692", negative="#ef6b73")
    ui.dark_mode(value=True)
    with ui.element("div").classes("sam-shell row no-wrap"):
        with ui.element("aside").classes("sam-sidebar column"):
            ui.html('<div class="sam-wordmark"><strong>SAM</strong>cloud</div>')
            for label, path, icon in _NAVIGATION:
                classes = "sam-nav sam-nav-active" if path == current_path else "sam-nav"
                ui.button(label, icon=icon, on_click=lambda destination=path: ui.navigate.to(destination)).props("flat no-caps").classes(classes)
            ui.label("COLMAP · SAM3").classes("sam-sidebar-footer")
        with ui.element("main").classes("sam-main"):
            ui.element("div").classes("sam-topbar")
            with ui.element("section").classes("sam-page"):
                ui.label(title).classes("sam-page-title")
                content()
