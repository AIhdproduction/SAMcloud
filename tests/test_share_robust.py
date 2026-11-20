"""Share viewer robustness on phones: device profile, crash guard, context loss, error details panel."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from samcloud.share import export

TEMPLATE = export.TEMPLATE_DIRECTORY

_HARNESS = r"""
const fs = require("fs");
function makeEnv({coarse, depthTexture, pixelRatio, fragDepth = true, ua = "TestAgent/1.0", maxTouch = 0}) {
  const calls = {budget: [], edl: [], pixelRatio: [], notify: [], crash: [], update: 0, recorded: 0};
  const listeners = {}, canvasListeners = {};
  const gl = { VERSION: 1, SHADING_LANGUAGE_VERSION: 2, MAX_TEXTURE_SIZE: 3, MAX_VERTEX_ATTRIBS: 4,
    getParameter(k) { return {1: "WebGL 1.0 (test)", 2: "GLSL ES 1.0", 3: 4096, 4: 16, 37446: "Mali-Test", 37445: "ARM"}[k]; },
    getExtension(n) {
      if (n === "WEBGL_debug_renderer_info") return {UNMASKED_RENDERER_WEBGL: 37446, UNMASKED_VENDOR_WEBGL: 37445};
      if (n === "WEBGL_depth_texture") return depthTexture ? {} : null;
      if (n === "EXT_frag_depth") return fragDepth ? {} : null;
      return n === "OES_element_index_uint" ? {} : null; } };
  const canvas = { width: 750, height: 1518, clientWidth: 375, clientHeight: 759,
    addEventListener(t, f) { canvasListeners[t] = f; } };
  let throwInUpdate = false;
  const viewer = { renderer: { domElement: canvas, getContext: () => gl, setPixelRatio: v => calls.pixelRatio.push(v) },
    setPointBudget: v => calls.budget.push(v), setEDLEnabled: v => calls.edl.push(v), getEDLEnabled: () => calls.edl[calls.edl.length-1],
    onCrash: e => calls.crash.push(e.message), update: () => { calls.update++; if (throwInUpdate) throw new Error("update boom"); } };
  const win = { matchMedia: () => ({matches: coarse}), screen: {width: 412, height: 915}, devicePixelRatio: pixelRatio,
    navigator: {userAgent: ua, maxTouchPoints: maxTouch},
    Potree: {PointShape: {SQUARE: 0, CIRCLE: 1, PARABOLOID: 2}}, addEventListener(t, f) { listeners[t] = f; } };
  if (coarse) win.ontouchstart = null;
  return {calls, listeners, canvasListeners, viewer, win, setThrow: v => { throwInUpdate = v; }};
}
global.window = {}; global.document = {getElementById: () => ({})}; global.setTimeout = () => 0;
eval(fs.readFileSync(process.argv[2], "utf8"));
const install = window.samcloudInstallRobustness;
const out = {};

let e = makeEnv({coarse: false, depthTexture: true, pixelRatio: 3});
let r = install(e.viewer, {notify: m => e.calls.notify.push(m), errorRecorded: () => e.calls.recorded++}, e.win);
out.desktop = {budget: e.calls.budget, edl: e.calls.edl, pixelRatio: e.calls.pixelRatio};
e = makeEnv({coarse: false, depthTexture: false, pixelRatio: 1});
install(e.viewer, {}, e.win);
out.desktopNoDepth = {edl: e.calls.edl};

e = makeEnv({coarse: true, depthTexture: true, pixelRatio: 3});
const hooks = {notify: m => e.calls.notify.push(m), errorRecorded: () => e.calls.recorded++};
r = install(e.viewer, hooks, e.win);
out.mobile = {budget: e.calls.budget.slice(), edl: e.calls.edl.slice(), pixelRatio: e.calls.pixelRatio.slice()};
e.viewer.onCrash(new Error("first"));
e.viewer.onCrash(new Error("second"));
out.afterTwo = {budget: e.calls.budget[e.calls.budget.length - 1], edl: e.calls.edl[e.calls.edl.length - 1],
  original: e.calls.crash.slice(), notify: e.calls.notify.slice(), errors: e.win.samcloudShareViewerErrors.map(x => x.message)};
e.viewer.onCrash(new Error("third"));
out.afterThree = {original: e.calls.crash.slice()};

// update() exceptions must not escape (the animation loop would stop being scheduled)
e.setThrow(true);
let escaped = false;
try { e.viewer.update(0.016, 1); } catch (err) { escaped = true; }
out.update = {escaped, recordedUpdate: e.win.samcloudShareViewerErrors.some(x => x.message === "update boom")};

// context loss / restore and global handlers
e = makeEnv({coarse: true, depthTexture: true, pixelRatio: 2});
r = install(e.viewer, {notify: m => e.calls.notify.push(m), errorRecorded: () => e.calls.recorded++}, e.win);
let prevented = false;
e.canvasListeners.webglcontextlost({preventDefault() { prevented = true; }});
const budgetBefore = e.calls.budget[e.calls.budget.length - 1];
e.canvasListeners.webglcontextrestored({});
out.context = {prevented, lostNotice: e.calls.notify[0], restoredNotice: e.calls.notify[1],
  budgetBefore, budgetAfter: e.calls.budget[e.calls.budget.length - 1]};
e.listeners.error({error: new Error("window failure")});
e.listeners.error({});  // resource errors without details are ignored
e.listeners.unhandledrejection({reason: new Error("worker rejected")});
const kinds = e.win.samcloudShareViewerErrors.map(x => x.kind + ":" + x.message);
out.global = {kinds, recorded: e.calls.recorded};
const text = r.debugInfo();
out.debug = {text};
// iOS: smaller budget (iPhone UA and iPadOS with a desktop Macintosh UA)
const iosBudget = (ua, maxTouch, coarse = true) => { const x = makeEnv({coarse, depthTexture: true, pixelRatio: 3, ua, maxTouch});
  install(x.viewer, {}, x.win); return x.calls.budget[0]; };
out.ios = {iphone: iosBudget("Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X)", 5),
  ipados: iosBudget("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)", 5),
  mac: iosBudget("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)", 0, false), android: iosBudget("Mozilla/5.0 (Linux; Android 10)", 5)};

// WebGL1 without EXT_frag_depth: the pick shader must use CIRCLE (PARABOLOID is remapped); with the extension it stays
const noExt = makeEnv({coarse: true, depthTexture: true, pixelRatio: 2, fragDepth: false});
install(noExt.viewer, {}, noExt.win);
const withExt = makeEnv({coarse: true, depthTexture: true, pixelRatio: 2, fragDepth: true});
install(withExt.viewer, {}, withExt.win);
out.pickShape = {noExt: noExt.win.Potree.PointShape.PARABOLOID, withExt: withExt.win.Potree.PointShape.PARABOLOID};

// shader-compile errors: picking disabled, budget reduced at most once, never the stock crash page, still logged
const sh = makeEnv({coarse: true, depthTexture: true, pixelRatio: 2, fragDepth: false});
const shRobust = install(sh.viewer, {notify: m => sh.calls.notify.push(m)}, sh.win);
for (let i = 0; i < 6; i++) sh.viewer.onCrash(new Error("could not compile shader pointcloud: 'GL_EXT_frag_depth' : extension is not supported"));
out.shader = {budgets: sh.calls.budget.slice(), original: sh.calls.crash.length, logged: shRobust.errors.length,
  pickingDisabled: sh.win.samcloudShareFlags.pickingDisabled};
console.log(JSON.stringify(out));
"""


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_robustness_profile_crash_guard_context_loss_and_error_log(tmp_path: Path) -> None:
    harness = tmp_path / "harness.js"
    harness.write_text(_HARNESS, encoding="utf-8")
    result = subprocess.run(["node", str(harness), str(TEMPLATE / "js" / "share_viewer.js")],
                            capture_output=True, text=True, encoding="utf-8", timeout=60)
    assert result.returncode == 0, result.stderr
    out = json.loads(result.stdout.strip().splitlines()[-1])
    assert out["desktop"] == {"budget": [3000000], "edl": [True], "pixelRatio": []}
    assert out["desktopNoDepth"]["edl"] == [False], "no depth texture: EDL off even on desktop"
    assert out["mobile"] == {"budget": [1000000], "edl": [False], "pixelRatio": [1.5]}
    after_two = out["afterTwo"]
    assert after_two["budget"] == 250000 and after_two["edl"] is False, "budget halves with a floor of 250k"
    assert after_two["original"] == [], "the stock crash page is not shown for the first two crashes"
    assert after_two["errors"] == ["first", "second"]
    assert after_two["notify"] == ["Darstellungsfehler – Punktdichte reduziert"] * 2
    assert out["afterThree"]["original"] == ["third"], "3 crashes within 10 s fall back to the original crash page"
    assert out["update"] == {"escaped": False, "recordedUpdate": True}
    context = out["context"]
    assert context["prevented"] is True and context["lostNotice"] == "Grafikkontext verloren – wird wiederhergestellt"
    assert "wiederhergestellt" in context["restoredNotice"] and context["budgetAfter"] < context["budgetBefore"]
    assert out["global"]["kinds"] == ["webglcontextlost:WebGL context lost", "window.onerror:window failure",
                                      "unhandledrejection:worker rejected"]
    assert out["global"]["recorded"] == 3
    assert out["ios"] == {"iphone": 700000, "ipados": 700000, "mac": 3000000, "android": 1000000}
    assert out["pickShape"] == {"noExt": 1, "withExt": 2}, "no EXT_frag_depth: pick shape forced to CIRCLE"
    shader = out["shader"]
    assert shader["budgets"] == [1000000, 500000], "shader errors reduce the budget only once"
    assert shader["original"] == 0 and shader["logged"] == 6 and shader["pickingDisabled"] is True
    text = out["debug"]["text"]
    for needle in ("window failure", "worker rejected", "userAgent: TestAgent/1.0", "devicePixelRatio: 2",
                   "canvas: 750x1518 (css 375x759)", "WebGL VERSION: WebGL 1.0 (test)", "SHADING_LANGUAGE_VERSION: GLSL ES 1.0",
                   "UNMASKED_RENDERER: Mali-Test", "UNMASKED_VENDOR: ARM", "MAX_TEXTURE_SIZE: 4096", "MAX_VERTEX_ATTRIBS: 16",
                   "EXT_frag_depth: yes", "WEBGL_depth_texture: yes", "OES_texture_float: no", "OES_element_index_uint: yes",
                   "point budget:", "EDL: false"):
        assert needle in text, needle


def test_error_details_panel_markup_and_wiring() -> None:
    html = (TEMPLATE / "index.html").read_text(encoding="utf-8")
    assert re.search(r'<button[^>]*id="error-button"[^>]*hidden', html), "badge hidden until an error is caught"
    assert 'id="error-panel"' in html and 'id="error-copy"' in html and 'id="error-close"' in html
    css = (TEMPLATE / "css" / "share.css").read_text(encoding="utf-8")
    legend_z = int(re.search(r"\.share-legend\s*\{[^}]*z-index:\s*(\d+)", css).group(1))
    panel_z = int(re.search(r"\.share-error-panel\s*\{[^}]*z-index:\s*(\d+)", css).group(1))
    assert panel_z > legend_z and "overflow: auto" in re.search(r"\.share-error-panel pre\s*\{([^}]*)\}", css).group(1)
    js = (TEMPLATE / "js" / "share_viewer.js").read_text(encoding="utf-8")
    assert "installRobustness(viewer" in js and "navigator.clipboard.writeText(text)" in js
    assert "setPointBudget(3000000)" not in js, "the budget comes from the device profile"
    assert 'addEventListener("unhandledrejection"' in js and 'addEventListener("error"' in js
    assert "webglcontextlost" in js and "webglcontextrestored" in js and "event.preventDefault()" in js


def test_ios_safari_page_gesture_and_scroll_blocking() -> None:
    js = (TEMPLATE / "js" / "share_viewer.js").read_text(encoding="utf-8")
    for type_ in ("gesturestart", "gesturechange", "gestureend"):
        assert f'"{type_}"' in js
    assert "{ passive: false }" in js and "blockPageZoom(document)" in js
    assert "maxTouchPoints" in js and "/Macintosh/" in js and "profile.ios ? 700000" in js
    assert "pointerdown" not in js and ".force" not in js and "radiusX" not in js, "plain touch events only"
    css = (TEMPLATE / "css" / "share.css").read_text(encoding="utf-8")
    assert re.search(r"html, body \{[^}]*position: fixed[^}]*overflow: hidden[^}]*overscroll-behavior: none", css)
    assert re.search(r"\.share-app \{[^}]*height: 100vh; height: 100dvh;", css)
    area = re.search(r"\.share-area, \.share-area \* \{([^}]*)\}", css).group(1)
    for rule in ("touch-action: none", "-webkit-user-select: none", "-webkit-touch-callout: none"):
        assert rule in area
    assert "-webkit-tap-highlight-color: transparent" in css


def test_pick_guard_is_in_the_touch_controller() -> None:
    js = (TEMPLATE / "js" / "share_viewer.js").read_text(encoding="utf-8")
    assert "if (flags.pickingDisabled) return null;" in js and "flags.pickingDisabled = true;" in js
    assert 'win.Potree.PointShape.PARABOLOID = win.Potree.PointShape.CIRCLE' in js
    assert "EXT_frag_depth" in js and "could not compile shader" in js
