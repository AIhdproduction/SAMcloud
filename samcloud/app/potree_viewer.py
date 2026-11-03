"""Potree-based point cloud viewer with the same controls as the canvas preview viewer."""

from __future__ import annotations

import json

from nicegui import ui

_LIB = "/assets-static/potree"

_HEAD = f"""
<link rel="stylesheet" type="text/css" href="{_LIB}/build/potree/potree.css">
<link rel="stylesheet" type="text/css" href="{_LIB}/libs/jquery-ui/jquery-ui.min.css">
<link rel="stylesheet" type="text/css" href="{_LIB}/libs/spectrum/spectrum.css">
<script src="{_LIB}/libs/jquery/jquery-3.1.1.min.js"></script>
<script src="{_LIB}/libs/spectrum/spectrum.js"></script>
<script src="{_LIB}/libs/jquery-ui/jquery-ui.min.js"></script>
<script src="{_LIB}/libs/other/BinaryHeap.js"></script>
<script src="{_LIB}/libs/tween/tween.min.js"></script>
<script src="{_LIB}/libs/d3/d3.js"></script>
<script src="{_LIB}/libs/proj4/proj4.js"></script>
<script src="{_LIB}/libs/i18next/i18next.js"></script>
<script src="{_LIB}/build/potree/potree.js"></script>
<style>
.sam-potree {{ position:relative; width:100%; height:calc(100vh - 150px); min-height:480px; overflow:hidden; border:1px solid #28475b; border-radius:9px; background:#050b12; }}
.sam-potree:fullscreen {{ width:100vw; height:100vh; border:0; border-radius:0; }}
.sam-potree-area {{ position:absolute; inset:0; }}
.sam-potree-toolbar {{ position:absolute; z-index:20; top:12px; left:12px; right:12px; display:flex; flex-wrap:wrap; gap:8px; align-items:center; pointer-events:none; }}
.sam-potree-toolbar > * {{ pointer-events:auto; }}
.sam-potree-toolbar button {{ border:1px solid #47718a; border-radius:5px; padding:6px 10px; background:#102131; color:#dcecf6; cursor:pointer; }}
.sam-potree-toolbar button.is-selected {{ border-color:#26b7ef; background:#16516c; }}
.sam-potree-toolbar label {{ display:flex; align-items:center; gap:6px; padding:5px 9px; border:1px solid #28475b; border-radius:5px; background:rgba(9,19,29,.86); color:#b9ccda; font-size:12px; }}
.sam-potree-status {{ position:absolute; z-index:20; left:12px; bottom:12px; padding:6px 9px; border-radius:5px; background:rgba(9,19,29,.86); color:#b9ccda; font-size:12px; }}
.sam-potree-classes {{ position:absolute; z-index:20; top:64px; right:12px; max-height:calc(100% - 90px); overflow:auto; width:min(300px,32vw); padding:10px; border:1px solid #365c74; border-radius:8px; background:rgba(9,19,29,.94); color:#eaf2f8; font-size:12px; display:none; }}
.sam-potree-classes.is-open {{ display:block; }}
.sam-potree-classes label {{ display:flex; align-items:center; gap:7px; padding:3px 0; cursor:pointer; }}
.sam-potree-classes i {{ width:12px; height:12px; border-radius:2px; display:inline-block; }}
.sam-potree .annotation-titlebar {{ font-size:14px; font-weight:700; padding:3px 9px; border:2px solid #ffffff; border-radius:6px; color:#06202f; box-shadow:0 0 0 2px rgba(0,0,0,.55); }}
.sam-potree .annotation {{ opacity:1 !important; }}
.sam-potree .annotation-titlebar {{ color:#06202f !important; }}
.sam-potree .annotation.is-hidden {{ display:none !important; }}
</style>
"""


def render_potree_viewer(
    kind: str,
    *,
    title: str,
    class_colors: list[list[int]],
    class_labels: list[str] | None = None,
    marker_endpoint: str | None = None,
) -> None:
    """Render a Potree viewer for the active project's sparse or dense cloud."""
    element_id = f"sam-potree-{kind}"
    config = {
        "element_id": element_id,
        "kind": kind,
        "title": title,
        "metadata_url": f"/api/potree/{kind}/metadata.json",
        "class_colors": class_colors,
        "class_labels": class_labels or [],
        "marker_endpoint": marker_endpoint,
        "show_classes": kind == "dense",
        "show_markers": bool(marker_endpoint),
    }
    ui.add_head_html(_HEAD)
    ui.html(
        _html(
            element_id,
            show_classes=config["show_classes"],
            show_markers=config["show_markers"],
        ),
        sanitize=False,
    ).classes("w-full")
    ui.add_body_html(_script(config))


def _html(
    element_id: str,
    *,
    show_classes: bool = True,
    show_markers: bool = False,
) -> str:
    markers_btn = (
        '    <button data-action="markers" class="is-selected">Passpunkte</button>\n'
        if show_markers
        else ""
    )
    classes_btns = (
        '    <button data-action="rgb" class="is-selected">RGB</button>\n'
        '    <button data-action="classification">Klassen</button>\n'
        '    <button data-action="classes">Klassen ein/aus</button>\n'
        if show_classes
        else ""
    )
    return f"""
<div id="{element_id}" class="sam-potree">
  <div class="sam-potree-area"></div>
  <div class="sam-potree-toolbar">
    <button data-action="fit">Alles zeigen</button>
{markers_btn}{classes_btns}    <button data-action="earth" class="is-selected">Terrain</button>
    <button data-action="orbit">Orbit</button>
    <button data-action="edl" class="is-selected">Schattierung</button>
    <button data-action="fullscreen">Vollbild</button>
    <label>Punktgrösse <input data-action="size" type="range" min="0.5" max="6" step="0.25" value="1.5"></label>
  </div>
  <div class="sam-potree-classes"></div>
  <div class="sam-potree-status">Punktwolke wird vorbereitet …</div>
</div>
"""


def _script(config: dict) -> str:
    return """
<script type="module">
import * as THREE from "/assets-static/potree/libs/three.js/build/three.module.js";
const config = __CONFIG__;
const start = () => {
  const root = document.getElementById(config.element_id);
  if (!root || root.dataset.ready === "1" || typeof Potree === "undefined") { setTimeout(start, 200); return; }
  root.dataset.ready = "1";
  const status = root.querySelector(".sam-potree-status");
  const viewer = new Potree.Viewer(root.querySelector(".sam-potree-area"));
  viewer.setEDLEnabled(true);
  viewer.setFOV(60);
  viewer.setPointBudget(3000000);
  viewer.setBackground("black");
  // Left drag rotates around the point under the cursor, right drag pans (same as the previous SAMcloud viewer).
  const earth = viewer.earthControls;
  const swapButtons = handler => event => {
    const drag = event.drag;
    if (!drag) return handler(event);
    const original = drag.mouse;
    if (original === Potree.MOUSE.LEFT) drag.mouse = Potree.MOUSE.RIGHT;
    else if (original === Potree.MOUSE.RIGHT) drag.mouse = Potree.MOUSE.LEFT;
    try { return handler(event); } finally { drag.mouse = original; }
  };
  for (const type of ["drag", "drop"]) {
    earth._listeners[type] = (earth._listeners[type] || []).map(swapButtons);
  }
  viewer.setControls(earth);
  const scheme = {};
  config.class_colors.forEach((colour, index) => {
    scheme[index] = { visible: true, name: config.class_labels[index] || `Klasse ${index}`, color: [colour[0] / 255, colour[1] / 255, colour[2] / 255, 1.0] };
  });
  scheme.DEFAULT = { visible: true, name: "Andere", color: [0.7, 0.7, 0.7, 1.0] };
  viewer.setClassifications(scheme);
  const panel = root.querySelector(".sam-potree-classes");
  panel.innerHTML = Object.keys(scheme).filter(key => key !== "DEFAULT").map(key => {
    const c = scheme[key].color.map(v => Math.round(v * 255));
    return `<label><input type="checkbox" data-class="${key}" checked><i style="background:rgb(${c[0]},${c[1]},${c[2]})"></i>${scheme[key].name}</label>`;
  }).join("");
  panel.addEventListener("change", event => {
    const key = event.target.dataset.class;
    if (key === undefined) return;
    scheme[key].visible = event.target.checked;
    viewer.setClassifications(scheme);
  });
  let cloud = null;
  const select = (actions, active) => actions.forEach(name => { const button = root.querySelector(`[data-action="${name}"]`); if (button) button.classList.toggle("is-selected", name === active); });
  root.querySelector(".sam-potree-toolbar").addEventListener("click", event => {
    const action = event.target.dataset && event.target.dataset.action;
    if (!action) return;
    if (action === "fit") viewer.fitToScreen(0.55);
    if (action === "markers") {
      markersVisible = !markersVisible;
      for (const item of markerObjects) {
        item.sphere.visible = markersVisible;
        const element = item.annotation.domElement && item.annotation.domElement[0];
        if (element) element.classList.toggle("is-hidden", !markersVisible);
      }
      event.target.classList.toggle("is-selected", markersVisible);
    }
    if ((action === "rgb" || action === "classification") && cloud) {
      cloud.material.activeAttributeName = action === "rgb" ? "rgba" : "classification";
      select(["rgb", "classification"], action);
    }
    if (action === "classes") panel.classList.toggle("is-open");
    if (action === "earth") { viewer.setControls(viewer.earthControls); select(["earth", "orbit"], "earth"); }
    if (action === "orbit") { viewer.setControls(viewer.orbitControls); select(["earth", "orbit"], "orbit"); }
    if (action === "edl") { viewer.setEDLEnabled(!viewer.getEDLEnabled()); event.target.classList.toggle("is-selected", viewer.getEDLEnabled()); }
    if (action === "fullscreen") { document.fullscreenElement ? document.exitFullscreen() : root.requestFullscreen(); }
  });
  root.querySelector('[data-action="size"]').addEventListener("input", event => { if (cloud) cloud.material.size = Number(event.target.value); });
  Potree.loadPointCloud(config.metadata_url, config.kind).then(event => {
    cloud = event.pointcloud;
    viewer.scene.addPointCloud(cloud);
    cloud.material.size = 1.5;
    cloud.material.pointSizeType = Potree.PointSizeType.ADAPTIVE;
    cloud.material.activeAttributeName = "rgba";
    viewer.fitToScreen(0.55);
    setTimeout(() => viewer.fitToScreen(0.55), 1500);
    fetch(config.metadata_url, { cache: "no-store" }).then(response => response.json()).then(metadata => {
      const count = Number(metadata.points || 0).toLocaleString("de-CH");
      status.textContent = `${config.title}: ${count} Punkte. Links ziehen: um den angeklickten Punkt drehen, rechts ziehen: verschieben, Mausrad: Zoom, Doppelklick: auf Punkt zoomen.`;
    }).catch(() => { status.textContent = `${config.title}. Links ziehen: um den angeklickten Punkt drehen, rechts ziehen: verschieben, Mausrad: Zoom, Doppelklick: auf Punkt zoomen.`; });
    loadMarkers();
  }).catch(error => { status.textContent = `Punktwolke nicht verfügbar: ${error && error.message ? error.message : error}`; });
  const colours = { pass: "#26b7ef", control: "#ffb020", disabled: "#6b7c8a", needs_review: "#ff5a5a" };
  const markerObjects = [];
  let markersVisible = true;
  viewer.addEventListener("update", () => {
    const camera = viewer.scene.getActiveCamera();
    for (const item of markerObjects) {
      const distance = camera.position.distanceTo(item.sphere.position);
      item.sphere.scale.setScalar(Math.max(0.5, distance * 0.012 / 0.18));
    }
  });
  const loadMarkers = async () => {
    if (!config.marker_endpoint) return;
    try {
      const response = await fetch(config.marker_endpoint, { cache: "no-store" });
      if (!response.ok) return;
      const payload = await response.json();
      for (const marker of payload.markers || []) {
        if (!Array.isArray(marker.coordinates) || marker.coordinates.length !== 3) continue;
        const annotation = new Potree.Annotation({ position: new THREE.Vector3(...marker.coordinates), title: marker.name });
        viewer.scene.annotations.add(annotation);
        const colour = colours[marker.state_color] || colours.needs_review;
        const sphere = new THREE.Mesh(
          new THREE.SphereGeometry(0.18, 20, 14),
          new THREE.MeshBasicMaterial({ color: colour, depthTest: false, transparent: true, opacity: 0.95 })
        );
        sphere.position.set(...marker.coordinates);
        sphere.renderOrder = 999;
        viewer.scene.scene.add(sphere);
        markerObjects.push({ annotation, sphere });
        const element = annotation.domElement && annotation.domElement[0];
        if (element) {
          element.style.setProperty("--sam-marker", colour);
          element.title = "Doppelklick: in der Passpunkt-Prüfung öffnen";
          const titlebar = element.querySelector(".annotation-titlebar");
          if (titlebar) { titlebar.style.background = colour; }
          element.addEventListener("click", () => window.dispatchEvent(new CustomEvent("samcloud-marker-selected", { detail: { name: marker.name } })));
          element.addEventListener("dblclick", event => {
            event.preventDefault();
            event.stopPropagation();
            window.location.href = `/control-points?marker=${encodeURIComponent(marker.name)}`;
          });
        }
      }
    } catch (error) { console.warn("Marker konnten nicht geladen werden", error); }
  };
};
start();
</script>
""".replace("__CONFIG__", json.dumps(config))
