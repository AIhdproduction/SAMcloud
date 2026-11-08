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
.sam-potree-clip-value {{ min-width:46px; text-align:right; font-variant-numeric:tabular-nums; color:#dcecf6; }}
.sam-potree-clip-value[data-clip-label="min"] {{ text-align:left; }}
.sam-potree-height {{ display:flex; align-items:center; gap:6px; padding:5px 9px; border:1px solid #28475b; border-radius:5px; background:rgba(9,19,29,.86); color:#b9ccda; font-size:12px; }}
.sam-potree-range {{ position:relative; width:180px; height:18px; }}
.sam-potree-range::before {{ content:""; position:absolute; left:0; right:0; top:7px; height:4px; border-radius:2px; background:#28475b; }}
.sam-potree-range-fill {{ position:absolute; top:7px; height:4px; border-radius:2px; background:#26b7ef; }}
.sam-potree-range input {{ position:absolute; left:0; top:0; width:100%; height:18px; margin:0; padding:0; background:none; pointer-events:none; -webkit-appearance:none; appearance:none; }}
.sam-potree-range input:focus {{ outline:none; }}
.sam-potree-range input::-webkit-slider-runnable-track {{ height:18px; background:transparent; border:0; }}
.sam-potree-range input::-moz-range-track {{ background:transparent; border:0; }}
.sam-potree-range input::-webkit-slider-thumb {{ -webkit-appearance:none; appearance:none; pointer-events:auto; width:14px; height:14px; margin-top:2px; border-radius:50%; background:#dcecf6; border:2px solid #26b7ef; cursor:ew-resize; }}
.sam-potree-range input::-moz-range-thumb {{ pointer-events:auto; width:10px; height:10px; border-radius:50%; background:#dcecf6; border:2px solid #26b7ef; cursor:ew-resize; }}
.sam-potree-status {{ position:absolute; z-index:20; left:12px; bottom:12px; padding:6px 9px; border-radius:5px; background:rgba(9,19,29,.86); color:#b9ccda; font-size:12px; }}
.sam-potree-classes {{ position:absolute; z-index:20; top:64px; right:12px; max-height:calc(100% - 90px); overflow:auto; width:min(300px,32vw); padding:10px; border:1px solid #365c74; border-radius:8px; background:rgba(9,19,29,.94); color:#eaf2f8; font-size:12px; display:none; }}
.sam-potree-classes.is-open {{ display:block; }}
.sam-potree-classes label {{ display:flex; align-items:center; gap:7px; padding:3px 0; cursor:pointer; }}
.sam-potree-classes i {{ width:12px; height:12px; border-radius:2px; display:inline-block; }}
.sam-potree .annotation-titlebar {{ font-size:14px; font-weight:700; padding:3px 9px; border:2px solid #ffffff; border-radius:6px; color:#06202f; box-shadow:0 0 0 2px rgba(0,0,0,.55); }}
.sam-potree .annotation {{ opacity:1 !important; }}
.sam-potree .annotation-titlebar {{ color:#06202f !important; }}
.sam-potree .annotation.is-hidden {{ display:none !important; }}
.sam-potree-camera-card {{ position:absolute; z-index:25; right:12px; bottom:12px; width:min(320px,42vw); padding:10px; border:1px solid #47718a; border-radius:7px; background:rgba(9,19,29,.94); color:#eaf2f8; font-size:12px; display:none; }}
.sam-potree-camera-card.is-open {{ display:block; }}
.sam-potree-camera-card img {{ display:block; width:100%; max-height:210px; margin-top:8px; object-fit:contain; background:#050b12; cursor:zoom-in; }}
.sam-potree-camera-card a {{ color:#59c8f3; display:inline-block; margin-top:7px; }}
</style>
"""


def render_potree_viewer(
    kind: str,
    *,
    title: str,
    class_colors: list[list[int]],
    class_labels: list[str] | None = None,
    marker_endpoint: str | None = None,
    camera_endpoint: str | None = None,
) -> None:
    """Render a Potree viewer for the active project's sparse or dense cloud."""
    config = _viewer_config(kind, title, class_colors, class_labels, marker_endpoint, camera_endpoint)
    element_id = config["element_id"]
    ui.add_head_html(_HEAD)
    ui.html(
        _html(
            element_id,
            show_classes=config["show_classes"],
            show_markers=config["show_markers"],
            show_cameras=config["show_cameras"],
        ),
        sanitize=False,
    ).classes("w-full")
    ui.add_body_html(_script(config))


def _viewer_config(
    kind: str,
    title: str,
    class_colors: list[list[int]],
    class_labels: list[str] | None = None,
    marker_endpoint: str | None = None,
    camera_endpoint: str | None = None,
) -> dict:
    """Build the browser configuration for one Potree cloud."""
    return {
        "element_id": f"sam-potree-{kind}",
        "kind": kind,
        "title": title,
        "metadata_url": f"/api/potree/{kind}/metadata.json",
        "status_url": f"/api/potree/{kind}/status",
        "class_colors": class_colors,
        "class_labels": class_labels or [],
        "marker_endpoint": marker_endpoint,
        "camera_endpoint": camera_endpoint,
        "show_classes": kind == "dense",
        "show_markers": bool(marker_endpoint),
        "show_cameras": bool(camera_endpoint),
    }


def _html(
    element_id: str,
    *,
    show_classes: bool = True,
    show_markers: bool = False,
    show_cameras: bool = False,
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
    cameras_btns = (
        '    <button data-action="cameras" class="is-selected">Kamerapositionen</button>\n'
        '    <button data-action="camera-colour">Reprojektionsfehler</button>\n'
        '    <button data-action="camera-sphere-texture" style="display:none">Panorama auf Kugel</button>\n'
        if show_cameras
        else ""
    )
    camera_sliders = (
        '    <label>Kameragrösse <input data-action="camera-size" type="range" min="0.25" max="0.75" step="0.25" value="0.5"></label>\n'
        if show_cameras
        else ""
    )
    return f"""
<div id="{element_id}" class="sam-potree">
  <div class="sam-potree-area"></div>
  <div class="sam-potree-toolbar">
    <button data-action="fit">Alles zeigen</button>
{markers_btn}{cameras_btns}{classes_btns}    <button data-action="earth" class="is-selected">Terrain</button>
    <button data-action="orbit">Orbit</button>
    <button data-action="edl" class="is-selected">Schattierung</button>
    <button data-action="fullscreen">Vollbild</button>
    <label>Punktgrösse <input data-action="size" type="range" min="0.5" max="6" step="0.25" value="1.5"></label>
    <div class="sam-potree-height" title="Höhenfilter: linker Knopf blendet Punkte unten aus, rechter Knopf oben">Höhe <span class="sam-potree-clip-value" data-clip-label="min">–</span><span class="sam-potree-range"><span class="sam-potree-range-fill"></span><input data-action="clip-min" type="range" min="0" max="1000" step="1" value="0" aria-label="Höhe unten"><input data-action="clip-max" type="range" min="0" max="1000" step="1" value="1000" aria-label="Höhe oben"></span><span class="sam-potree-clip-value" data-clip-label="max">–</span></div>
{camera_sliders}
  </div>
  <div class="sam-potree-classes"></div>
  <aside class="sam-potree-camera-card"></aside>
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
   const cameraSettingsKey = "samcloud:sparse-camera-settings:v1";
   const savedCameraSettings = (() => { try { return JSON.parse(localStorage.getItem(cameraSettingsKey) || "{}"); } catch (_error) { return {}; } })();
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
    if (action === "cameras") {
      camerasVisible = !camerasVisible;
      syncCameraSettings();
      saveCameraSettings();
    }
    if (action === "camera-sphere-texture") {
      cameraSphereTextures = !cameraSphereTextures;
      event.target.classList.toggle("is-selected", cameraSphereTextures);
      saveCameraSettings();
      syncSphereTextures();
      return;
    }
    if (action === "camera-colour") {
      cameraErrorColours = !cameraErrorColours;
      updateCameraColours();
      event.target.classList.toggle("is-selected", cameraErrorColours);
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
   /* Höhenfilter: ein Regler mit zwei Knöpfen (unten/oben) von tiefstem bis höchstem Punkt; alles ausserhalb wird über eine
      unsichtbare Potree-Clipbox ausgeblendet. Die Box wird nur in scene.volumes geführt (nicht in
      der three.js-Szene), damit sie weder sichtbar noch per Maus verschiebbar ist. */
   const clipMin = root.querySelector('[data-action="clip-min"]');
   const clipMax = root.querySelector('[data-action="clip-max"]');
   const clipFill = root.querySelector('.sam-potree-range-fill');
   let heightBounds = null;
   let clipVolume = null;
   const sliderHeight = value => heightBounds.min[2] + (heightBounds.max[2] - heightBounds.min[2]) * Number(value) / 1000;
   const applyHeightClip = changed => {
     if (!heightBounds) return;
     let low = Number(clipMin.value);
     let high = Number(clipMax.value);
     /* Die Knöpfe dürfen sich nicht überholen: der bewegte Knopf bleibt am anderen stehen. */
     if (low > high) {
       if (changed === clipMax) { clipMax.value = low; high = low; } else { clipMin.value = high; low = high; }
     }
     clipFill.style.left = `calc(7px + (100% - 14px) * ${low / 1000})`;
     clipFill.style.width = `calc((100% - 14px) * ${(high - low) / 1000})`;
     /* Liegen beide Knöpfe ganz rechts, muss der untere greifbar bleiben. */
     clipMin.style.zIndex = low > 990 ? "2" : "1";
     clipMax.style.zIndex = low > 990 ? "1" : "2";
     const zLow = sliderHeight(low);
     const zHigh = sliderHeight(high);
     root.querySelector('[data-clip-label="min"]').textContent = zLow.toFixed(2);
     root.querySelector('[data-clip-label="max"]').textContent = zHigh.toFixed(2);
     const active = low > 0 || high < 1000;
     if (!clipVolume) {
       clipVolume = new Potree.BoxVolume();
       clipVolume.visible = false;
       viewer.scene.volumes.push(clipVolume);
     }
     const [minX, minY] = heightBounds.min;
     const [maxX, maxY] = heightBounds.max;
     const margin = Math.max(maxX - minX, maxY - minY, 1);
     clipVolume.position.set((minX + maxX) / 2, (minY + maxY) / 2, (zLow + zHigh) / 2);
     clipVolume.scale.set(maxX - minX + margin, maxY - minY + margin, Math.max(zHigh - zLow, 1e-4));
     clipVolume.updateMatrixWorld();
     clipVolume.clip = active;
     viewer.setClipTask(active ? Potree.ClipTask.SHOW_INSIDE : Potree.ClipTask.NONE);
   };
   const initHeightClip = metadata => {
     const position = (metadata.attributes || []).find(attribute => attribute.name === "position");
     const bounds = position && Array.isArray(position.min) && Array.isArray(position.max) ? position : metadata.boundingBox;
     if (!bounds || !Array.isArray(bounds.min) || !Array.isArray(bounds.max)) return;
     heightBounds = { min: bounds.min.map(Number), max: bounds.max.map(Number) };
     applyHeightClip(null);
   };
   [clipMin, clipMax].forEach(control => control.addEventListener("input", event => applyHeightClip(event.target)));
   [["camera-size", "cameraSize"]].forEach(([action, key]) => {
     const control = root.querySelector(`[data-action="${action}"]`);
     if (control) control.addEventListener("input", event => {
       if (key === "cameraSize") cameraSize = Number(event.target.value);
       if (key === "cameraPathWidth") cameraPathWidth = Number(event.target.value);
       if (key === "cameraPhotoSize") cameraPhotoSize = Number(event.target.value);
       if (cameraPath) cameraPath.material.linewidth = cameraPathWidth;
       saveCameraSettings();
       syncCameraBillboards();
     });
   });
  const pageStarted = Date.now();
  let loadAttempts = 0;
  const formatElapsed = seconds => {
    const s = Math.max(0, Math.floor(seconds));
    return `${String(Math.floor(s / 60)).padStart(2, "0")}:${String(s % 60).padStart(2, "0")}`;
  };
  const showError = message => {
    status.innerHTML = "";
    status.append(`Punktwolke nicht verfügbar: ${message} `);
    const retry = document.createElement("button");
    retry.type = "button";
    retry.textContent = "Erneut versuchen";
    retry.className = "sam-potree-retry";
    retry.style.cssText = "margin-left:8px;padding:2px 8px;border-radius:4px;border:1px solid #3a5a74;background:#16324a;color:#e6f1f8;cursor:pointer;";
    retry.addEventListener("click", () => { loadAttempts = 0; pollStatus(); });
    status.append(retry);
  };
  const loadCloud = () => {
    Potree.loadPointCloud(config.metadata_url, config.kind).then(event => {
      cloud = event.pointcloud;
      viewer.scene.addPointCloud(cloud);
      cloud.material.size = 1.5;
      cloud.material.pointSizeType = Potree.PointSizeType.ADAPTIVE;
      cloud.material.activeAttributeName = "rgba";
      viewer.fitToScreen(0.55);
      setTimeout(() => viewer.fitToScreen(0.55), 1500);
      fetch(config.metadata_url, { cache: "no-store" }).then(response => response.json()).then(metadata => {
        initHeightClip(metadata);
        const count = Number(metadata.points || 0).toLocaleString("de-CH");
        status.textContent = `${config.title}: ${count} Punkte. Links ziehen: um den angeklickten Punkt drehen, rechts ziehen: verschieben, Mausrad: Zoom, Doppelklick: auf Punkt zoomen.`;
      }).catch(() => { status.textContent = `${config.title}. Links ziehen: um den angeklickten Punkt drehen, rechts ziehen: verschieben, Mausrad: Zoom, Doppelklick: auf Punkt zoomen.`; });
      loadMarkers();
      loadCameras();
    }).catch(error => {
      loadAttempts += 1;
      if (loadAttempts < 3) { setTimeout(pollStatus, 2000); return; }
      showError(error && error.message ? error.message : String(error));
    });
  };
  const pollStatus = async () => {
    let state;
    try {
      const response = await fetch(config.status_url, { cache: "no-store" });
      state = await response.json();
    } catch (error) {
      setTimeout(pollStatus, 2000);
      return;
    }
    if (state.state === "ready") { loadCloud(); return; }
    if (state.state === "error") { showError(state.message || "Fehler beim Vorbereiten"); return; }
    if (state.state === "missing" && state.message && !String(state.message).includes("not been prepared")) {
      status.textContent = `Punktwolke nicht verfügbar: ${state.message}`;
      return;
    }
    const startedMs = state.started_at ? state.started_at * 1000 : pageStarted;
    status.textContent = `Punktwolke wird vorbereitet … ${formatElapsed((Date.now() - startedMs) / 1000)}`;
    setTimeout(pollStatus, 2000);
  };
  pollStatus();
  const colours = { pass: "#26b7ef", control: "#ffb020", disabled: "#6b7c8a", needs_review: "#ff5a5a" };
  const markerObjects = [];
  let markersVisible = true;
   const cameraCard = root.querySelector(".sam-potree-camera-card");
   const cameraLayer = new THREE.Group();
   const cameraMarkerLayer = new THREE.Group();
   const cameraPhotoLayer = new THREE.Group();
   const cameraObjects = [];
   const cameraPhotos = new Map();
   const cameraPhotoLoads = new Set();
   const maxCameraPhotos = 72;
   let cameraPath = null;
   let camerasVisible = savedCameraSettings.positions !== false;
   let cameraPathVisible = false; /* Kamerapfad entfernt */
   let cameraPhotosVisible = true; /* Foto-Ansicht (Info-Karte) immer verfügbar */
   let cameraSize = Math.min(.75, Math.max(.25, Math.round((Number(savedCameraSettings.size) || .5) * 4) / 4));
   let cameraPathWidth = Number(savedCameraSettings.pathWidth) || 1.5;
   let cameraPhotoSize = Number(savedCameraSettings.photoSize) || 1;
   let cameraErrorColours = false;
   let cameraPanorama = false;
   let cameraSphereTextures = savedCameraSettings.sphereTextures === true;
   const sphereTextures = new Map();
   const sphereTextureLoads = new Set();
   const maxSphereTextures = 48;
  const raycaster = new THREE.Raycaster();
  const pointer = new THREE.Vector2();
   cameraLayer.add(cameraMarkerLayer);
   cameraLayer.add(cameraPhotoLayer);
   viewer.scene.scene.add(cameraLayer);
   const saveCameraSettings = () => {
     try { localStorage.setItem(cameraSettingsKey, JSON.stringify({ positions: camerasVisible, path: cameraPathVisible, photos: cameraPhotosVisible, size: cameraSize, pathWidth: cameraPathWidth, photoSize: cameraPhotoSize, sphereTextures: cameraSphereTextures })); } catch (_error) {}
   };
   const syncCameraSettings = () => {
     cameraMarkerLayer.visible = camerasVisible;
     cameraPhotoLayer.visible = cameraPhotosVisible;
     if (cameraPath) cameraPath.visible = cameraPathVisible;
     [["cameras", camerasVisible], ["camera-path", cameraPathVisible], ["camera-photos", cameraPhotosVisible]].forEach(([action, selected]) => { const control = root.querySelector(`[data-action="${action}"]`); if (control) control.classList.toggle("is-selected", selected); });
     [["camera-size", cameraSize], ["camera-path-width", cameraPathWidth], ["camera-photo-size", cameraPhotoSize]].forEach(([action, value]) => { const control = root.querySelector(`[data-action="${action}"]`); if (control) control.value = value; });
   };
   const textureLoader = new THREE.TextureLoader();
   const syncCameraBillboards = () => {};
   const syncSphereTextures = () => {
     if (!cameraPanorama || !cameraObjects.length) return;
     const activeCamera = viewer.scene.getActiveCamera();
     if (!cameraSphereTextures) {
       for (const item of cameraObjects) {
         if (item.marker.material.map) { item.marker.material.map = null; item.marker.material.needsUpdate = true; }
       }
       return;
     }
     const nearest = cameraObjects.map(item => ({ item, distance: activeCamera.position.distanceTo(item.marker.position) }))
       .sort((a, b) => a.distance - b.distance).slice(0, maxSphereTextures);
     const wanted = new Set(nearest.map(entry => entry.item.camera.id));
     for (const item of cameraObjects) {
       if (!wanted.has(item.camera.id) && item.marker.material.map) {
         const texture = sphereTextures.get(item.camera.id);
         item.marker.material.map = null; item.marker.material.needsUpdate = true;
         if (texture) { texture.dispose(); sphereTextures.delete(item.camera.id); }
       }
     }
     for (const { item } of nearest) {
       const id = item.camera.id;
       if (sphereTextures.has(id)) { if (item.marker.material.map !== sphereTextures.get(id)) { item.marker.material.map = sphereTextures.get(id); item.marker.material.color.set(0xffffff); item.marker.material.needsUpdate = true; } continue; }
       if (sphereTextureLoads.has(id)) continue;
       sphereTextureLoads.add(id);
       textureLoader.load(`/api/camera-thumbnail/${encodeURIComponent(id)}?size=640`, texture => {
         sphereTextureLoads.delete(id);
         if (!cameraSphereTextures) { texture.dispose(); return; }
         sphereTextures.set(id, texture);
         item.marker.material.map = texture; item.marker.material.color.set(0xffffff); item.marker.material.needsUpdate = true;
       }, undefined, () => sphereTextureLoads.delete(id));
     }
   };
   syncCameraSettings();
   viewer.addEventListener("update", () => {
     const camera = viewer.scene.getActiveCamera();
    for (const item of markerObjects) {
      const distance = camera.position.distanceTo(item.sphere.position);
      item.sphere.scale.setScalar(Math.max(0.5, distance * 0.012 / 0.18));
    }
     for (const item of cameraObjects) {
       const distance = camera.position.distanceTo(item.marker.position);
       item.marker.scale.setScalar(Math.max(0.35 * cameraSize, distance * 0.012 * cameraSize));
     }
     syncCameraBillboards();
     syncSphereTextures();
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
  const cameraColour = camera => {
    if (cameraErrorColours && Number.isFinite(camera.mean_reprojection_error)) {
      const error = Math.min(4, Math.max(0, Number(camera.mean_reprojection_error)));
      return new THREE.Color().setHSL((1 - error / 4) * .34, .86, .55);
    }
    const count = Math.max(1, cameraObjects.length - 1);
    return new THREE.Color().setHSL((Number(camera.index || 0) / count) * .72, .86, .55);
  };
  const updateCameraColours = () => {
    const pathColours = [];
    for (const item of cameraObjects) {
      const colour = cameraColour(item.camera);
      item.marker.material.color.copy(colour);
      pathColours.push(colour.r, colour.g, colour.b);
    }
    if (cameraPath && cameraPath.geometry.getAttribute("color")) {
      cameraPath.geometry.setAttribute("color", new THREE.Float32BufferAttribute(pathColours, 3));
    }
  };
   let cameraCardTimer = null;
   let cameraCardPinned = false;
   let cameraCardHovered = false;
   const cameraCardHoldMs = 5000;
   const armCameraCardTimer = () => {
     clearTimeout(cameraCardTimer);
     cameraCardTimer = setTimeout(() => {
       if (cameraCardHovered) return;
       cameraCardPinned = false;
       if (cameraCard) cameraCard.classList.remove("is-open");
     }, cameraCardHoldMs);
   };
   if (cameraCard) {
     cameraCard.addEventListener("mouseenter", () => { cameraCardHovered = true; clearTimeout(cameraCardTimer); });
     cameraCard.addEventListener("mouseleave", () => { cameraCardHovered = false; armCameraCardTimer(); });
   }
   const showCamera = (camera, includeImage) => {
     if (!cameraCard || !cameraPhotosVisible) return;
     /* Ein angeklicktes Foto bleibt 5 s unten rechts stehen; Hover auf andere Marker ersetzt es nicht. */
     if (!includeImage && cameraCardPinned) return;
     if (includeImage) cameraCardPinned = true;
    cameraCard.innerHTML = "";
    const title = document.createElement("strong");
    title.textContent = camera.source_name || camera.image_name || "Kamera";
    cameraCard.append(title);
    const details = document.createElement("div");
    const error = Number.isFinite(camera.mean_reprojection_error) ? `${Number(camera.mean_reprojection_error).toFixed(2)} px` : "–";
    details.textContent = `Aufnahme ${Number(camera.index) + 1} · ${camera.points3d || 0} 3D-Punkte · Reprojektionsfehler ${error}`;
    cameraCard.append(details);
    if (includeImage && camera.id) {
      const imageUrl = `/api/camera-thumbnail/${encodeURIComponent(camera.id)}?size=320`;
      const image = document.createElement("img");
      image.alt = camera.source_name || camera.image_name || "Kamerafoto";
      image.src = imageUrl;
      image.title = "Klicken: Foto gross öffnen";
      image.addEventListener("click", () => window.open(imageUrl.replace("size=320", "size=1024"), "_blank", "noopener"));
      cameraCard.append(image);
      const link = document.createElement("a");
      link.href = imageUrl;
      link.target = "_blank";
      link.rel = "noopener";
      link.textContent = "Foto anzeigen";
      cameraCard.append(link);
    }
    cameraCard.classList.add("is-open");
    armCameraCardTimer();
  };
   const pickCamera = (event, includeImage) => {
     if (!cameraObjects.length || !camerasVisible || !cameraPhotosVisible) return;
    const rect = root.getBoundingClientRect();
    pointer.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
    pointer.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
    raycaster.setFromCamera(pointer, viewer.scene.getActiveCamera());
    const hit = raycaster.intersectObjects(cameraObjects.map(item => item.marker), false)[0];
    if (hit && hit.object.userData.camera) showCamera(hit.object.userData.camera, includeImage);
  };
  const cameraAt = event => {
    if (!cameraObjects.length || !camerasVisible) return null;
    const rect = root.getBoundingClientRect();
    pointer.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
    pointer.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
    raycaster.setFromCamera(pointer, viewer.scene.getActiveCamera());
    const hit = raycaster.intersectObjects(cameraObjects.map(item => item.marker), false)[0];
    return hit && hit.object.userData.camera ? hit.object.userData.camera : null;
  };
  /* Doppelklick auf eine Kameraposition öffnet direkt das zugehörige Foto. */
  root.querySelector(".sam-potree-area").addEventListener("dblclick", event => {
    const camera = cameraAt(event);
    if (!camera || !camera.id) return;
    event.preventDefault();
    event.stopPropagation();
    window.open(`/api/camera-thumbnail/${encodeURIComponent(camera.id)}?size=1024`, "_blank", "noopener");
  }, true);
  root.querySelector(".sam-potree-area").addEventListener("pointermove", event => pickCamera(event, false));
  root.querySelector(".sam-potree-area").addEventListener("click", event => pickCamera(event, true));
  const loadCameras = async () => {
    if (!config.camera_endpoint || cameraObjects.length) return;
    try {
      const response = await fetch(config.camera_endpoint, { cache: "no-store" });
      if (!response.ok) return;
      const payload = await response.json();
      const cameras = (payload.cameras || []).filter(camera => Array.isArray(camera.position) && camera.position.length === 3);
      cameraPanorama = payload.is_panorama === true;
      const cone = new THREE.ConeGeometry(.65, 1.5, 4);
      const ball = new THREE.SphereGeometry(.7, 24, 16);
      const sphereButton = root.querySelector('[data-action="camera-sphere-texture"]');
      if (sphereButton) sphereButton.style.display = cameraPanorama ? "" : "none";
      for (const camera of cameras) {
        const marker = new THREE.Mesh(cameraPanorama ? ball : cone, new THREE.MeshBasicMaterial({ depthTest: true, depthWrite: true }));
        marker.position.set(...camera.position);
        const direction = new THREE.Vector3(...(camera.direction || [0, 0, 1]));
        if (!cameraPanorama && direction.lengthSq() > 0) marker.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), direction.normalize());
        marker.userData.camera = camera;
         cameraMarkerLayer.add(marker);
        cameraObjects.push({ camera, marker });
      }
       updateCameraColours();
       const sphereToggle = root.querySelector('[data-action="camera-sphere-texture"]');
       if (sphereToggle) sphereToggle.classList.toggle("is-selected", cameraSphereTextures);
       syncCameraSettings();
       syncCameraBillboards();
    } catch (error) { console.warn("Kamerapositionen konnten nicht geladen werden", error); }
  };
};
start();
</script>
""".replace("__CONFIG__", json.dumps(config))
