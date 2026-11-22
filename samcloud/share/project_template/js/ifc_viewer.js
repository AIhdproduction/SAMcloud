import * as THREE from "../lib/three/three.module.js";
import { OrbitControls } from "../lib/three/OrbitControls.js";
import { GLTFLoader } from "../lib/three/GLTFLoader.js";

const TEXT = {
  de: {
    title: "IFC-Modell",
    fit: "Einpassen",
    categories: "Kategorien",
    level: "Ebene",
    reset: "Ansicht zurücksetzen",
    all: "Alle",
    none: "Keine",
    errorDetails: "Fehlerdetails",
    copy: "Kopieren",
    close: "Schliessen",
    contextLost: "Grafik wurde zurückgesetzt.",
    reload: "Neu laden",
    loading: "IFC-Modell wird geladen … {percent} %",
    loadingView: "IFC-Ansicht wird geladen …",
    ready: "Bereit",
    noIfc: "Für diese Ebene ist kein IFC-Modell verfügbar.",
    loadFailed: "Das IFC-Modell konnte nicht geladen werden: {message}",
    projectFailed: "Die Projektinformationen konnten nicht geladen werden: {message}",
  },
  en: {
    title: "IFC model",
    fit: "Fit",
    categories: "Categories",
    level: "Level",
    reset: "Reset view",
    all: "All",
    none: "None",
    errorDetails: "Error details",
    copy: "Copy",
    close: "Close",
    contextLost: "Graphics were reset.",
    reload: "Reload",
    loading: "Loading IFC model … {percent} %",
    loadingView: "Loading IFC view …",
    ready: "Ready",
    noIfc: "No IFC model is available for this level.",
    loadFailed: "The IFC model could not be loaded: {message}",
    projectFailed: "The project information could not be loaded: {message}",
  },
};

const dom = {
  canvas: document.getElementById("ifc-canvas"),
  toolbar: document.getElementById("ifc-toolbar"),
  fit: document.getElementById("fit-button"),
  categories: document.getElementById("categories-button"),
  levelControl: document.getElementById("level-control"),
  levelSelect: document.getElementById("level-select"),
  reset: document.getElementById("reset-button"),
  legend: document.getElementById("category-legend"),
  status: document.getElementById("ifc-status"),
  errorButton: document.getElementById("error-button"),
  errorPanel: document.getElementById("error-panel"),
  errorText: document.getElementById("error-text"),
  errorCopy: document.getElementById("error-copy"),
  errorClose: document.getElementById("error-close"),
  contextOverlay: document.getElementById("context-overlay"),
  contextMessage: document.getElementById("context-message"),
  contextReload: document.getElementById("context-reload"),
};

const state = {
  language: "de",
  project: null,
  levels: [],
  currentLevel: null,
  view: null,
  renderer: null,
  scene: null,
  camera: null,
  controls: null,
  modelRoot: null,
  categoryMeshes: new Map(),
  categoryVisible: new Map(),
  materials: new Map(),
  initialView: null,
  drawRequested: false,
  loadToken: 0,
  errors: [],
  contextLost: false,
};

const text = key => TEXT[state.language][key] ?? TEXT.de[key] ?? key;
const replace = (key, values) => text(key).replace(/\{(\w+)\}/g, (_match, name) => String(values[name] ?? ""));
const own = (object, key) => Object.prototype.hasOwnProperty.call(object || {}, key);
const modelUrl = (path, file) => new URL(`${String(path || "").replace(/\/?$/, "/")}${file}`, window.location.href).href;
const formatCount = value => String(Math.round(Number(value || 0))).replace(/\B(?=(\d{3})+(?!\d))/g, "\u2009");

let statusTimer = 0;
function setStatus(message, autoHideMs = 0) {
  clearTimeout(statusTimer);
  dom.status.textContent = message || "";
  dom.status.hidden = !message;
  if (message && autoHideMs > 0) {
    statusTimer = setTimeout(() => {
      dom.status.hidden = true;
    }, autoHideMs);
  }
}

function recordError(kind, error) {
  const value = error instanceof Error ? error : new Error(String(error));
  state.errors.push({ kind, message: value.message || String(value), stack: value.stack || "", time: new Date().toISOString() });
  if (state.errors.length > 50) state.errors.shift();
  dom.errorButton.hidden = false;
  if (!dom.errorPanel.hidden) updateErrorPanel();
}

function diagnostics() {
  const lines = ["SAMcloud IFC share viewer error details", ""];
  for (const item of state.errors) lines.push(`[${item.time}] ${item.kind}: ${item.message}`, item.stack, "");
  if (!state.errors.length) lines.push("(no errors recorded)", "");
  lines.push(`userAgent: ${navigator.userAgent}`, `devicePixelRatio: ${window.devicePixelRatio || 1}`);
  if (state.renderer) {
    const canvas = state.renderer.domElement;
    lines.push(`canvas: ${canvas.width}x${canvas.height} (css ${canvas.clientWidth}x${canvas.clientHeight})`);
  }
  if (state.currentLevel) lines.push(`level: ${state.currentLevel.key}`);
  return lines.join("\n");
}

function updateErrorPanel() {
  dom.errorText.textContent = diagnostics();
}

function showError(error, statusKey) {
  recordError("viewer", error);
  setStatus(replace(statusKey, { message: error && error.message ? error.message : String(error) }));
}

function applyTexts() {
  document.documentElement.lang = state.language;
  document.title = `${state.project && state.project.name ? state.project.name : text("title")} – SAMcloud`;
  dom.fit.textContent = text("fit");
  dom.categories.textContent = text("categories");
  dom.categories.setAttribute("aria-label", text("categories"));
  dom.levelControl.querySelector("span").textContent = text("level");
  dom.levelSelect.setAttribute("aria-label", text("level"));
  dom.reset.textContent = text("reset");
  dom.legend.setAttribute("aria-label", text("categories"));
  dom.errorButton.textContent = text("errorDetails");
  dom.errorPanel.setAttribute("aria-label", text("errorDetails"));
  dom.errorCopy.textContent = text("copy");
  dom.errorClose.textContent = text("close");
  dom.contextMessage.textContent = text("contextLost");
  dom.contextReload.textContent = text("reload");
}

function copyDiagnostics() {
  const value = diagnostics();
  const fallback = () => {
    const area = document.createElement("textarea");
    area.value = value;
    document.body.append(area);
    area.select();
    try { document.execCommand("copy"); } catch (_error) { /* Clipboard access is optional. */ }
    area.remove();
  };
  if (navigator.clipboard && navigator.clipboard.writeText) navigator.clipboard.writeText(value).catch(fallback);
  else fallback();
}

function fetchJson(path) {
  return fetch(path, { cache: "no-store" }).then(response => {
    if (!response.ok) throw new Error(`${path}: ${response.status}`);
    return response.json();
  });
}

function projectFromWindow() {
  return window.SAMShare && window.SAMShare.project ? window.SAMShare.project : null;
}

function waitForProject() {
  const immediate = projectFromWindow();
  if (immediate) return Promise.resolve(immediate);
  return new Promise((resolve, reject) => {
    let settled = false;
    const finish = callback => value => {
      if (settled) return;
      settled = true;
      window.removeEventListener("samshare:project", onProject);
      callback(value);
    };
    const onProject = event => finish(resolve)(event.detail && event.detail.project ? event.detail.project : event.detail || projectFromWindow());
    window.addEventListener("samshare:project", onProject, { once: true });
    window.setTimeout(() => {
      if (settled) return;
      fetchJson("data/project.json").then(finish(resolve), finish(reject));
    }, 3000);
  });
}

function isPhoneProfile() {
  return (window.matchMedia && window.matchMedia("(pointer: coarse)").matches)
    || Math.min(window.innerWidth, window.innerHeight) < 700;
}

function setupScene() {
  const mobile = isPhoneProfile();
  state.renderer = new THREE.WebGLRenderer({ antialias: !mobile, alpha: false, powerPreference: mobile ? "low-power" : "high-performance" });
  state.renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, mobile ? 1.5 : 2));
  state.renderer.setSize(Math.max(1, dom.canvas.clientWidth), Math.max(1, dom.canvas.clientHeight), false);
  state.renderer.outputColorSpace = THREE.SRGBColorSpace;
  state.renderer.toneMapping = THREE.ACESFilmicToneMapping;
  state.renderer.toneMappingExposure = 1;
  state.renderer.domElement.setAttribute("aria-label", "IFC 3D viewer");
  dom.canvas.append(state.renderer.domElement);

  state.scene = new THREE.Scene();
  state.scene.background = new THREE.Color("#050b12");
  state.camera = new THREE.PerspectiveCamera(45, 1, 0.01, 1000);
  state.scene.add(new THREE.HemisphereLight(0xffffff, 0x203548, 1.3));
  const sun = new THREE.DirectionalLight(0xffffff, 1.8);
  sun.position.set(6, 10, 8);
  state.scene.add(sun);
  state.modelRoot = new THREE.Group();
  state.scene.add(state.modelRoot);

  state.controls = new OrbitControls(state.camera, state.renderer.domElement);
  state.controls.enableDamping = true;
  state.controls.dampingFactor = 0.08;
  state.controls.screenSpacePanning = true;
  state.controls.touches.ONE = THREE.TOUCH.ROTATE;
  state.controls.touches.TWO = THREE.TOUCH.DOLLY_PAN;
  state.controls.addEventListener("change", requestDraw);
  state.renderer.domElement.addEventListener("webglcontextlost", event => {
    event.preventDefault();
    state.contextLost = true;
    recordError("webglcontextlost", new Error("WebGL context lost"));
    dom.contextOverlay.hidden = false;
  }, false);
  state.renderer.domElement.addEventListener("webglcontextrestored", () => window.location.reload(), false);
  resize();
}

function requestDraw() {
  if (state.drawRequested || !state.renderer || state.contextLost) return;
  state.drawRequested = true;
  requestAnimationFrame(() => {
    state.drawRequested = false;
    if (!state.renderer || state.contextLost) return;
    state.controls.update();
    state.renderer.render(state.scene, state.camera);
  });
}

function resize() {
  if (!state.renderer) return;
  const width = Math.max(1, Math.round(dom.canvas.clientWidth));
  const height = Math.max(1, Math.round(dom.canvas.clientHeight));
  const mobile = isPhoneProfile();
  state.renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, mobile ? 1.5 : 2));
  state.renderer.setSize(width, height, false);
  state.camera.aspect = width / height;
  state.camera.updateProjectionMatrix();
  requestDraw();
}

function disposeMaterial(material, disposedMaterials, disposedTextures) {
  if (!material || disposedMaterials.has(material)) return;
  disposedMaterials.add(material);
  for (const value of Object.values(material)) {
    if (value && value.isTexture && !disposedTextures.has(value)) {
      disposedTextures.add(value);
      value.dispose();
    }
  }
  material.dispose();
}

function disposeCurrentModel() {
  state.loadToken += 1;
  const disposedGeometry = new Set();
  const disposedMaterials = new Set();
  const disposedTextures = new Set();
  if (state.modelRoot) {
    state.modelRoot.traverse(object => {
      if (object.geometry && !disposedGeometry.has(object.geometry)) {
        disposedGeometry.add(object.geometry);
        object.geometry.dispose();
      }
      if (object.material) {
        const materials = Array.isArray(object.material) ? object.material : [object.material];
        materials.forEach(material => disposeMaterial(material, disposedMaterials, disposedTextures));
      }
    });
    state.modelRoot.clear();
  }
  state.categoryMeshes.clear();
  state.categoryVisible.clear();
  state.materials.clear();
  state.view = null;
  state.initialView = null;
}

function categoryFor(object, byNode) {
  for (let node = object; node; node = node.parent) {
    if (node.name && own(byNode, node.name)) return byNode[node.name];
  }
  return "other";
}

function addCategoryMesh(category, mesh) {
  if (!state.categoryMeshes.has(category)) state.categoryMeshes.set(category, []);
  state.categoryMeshes.get(category).push(mesh);
}

function prepareModel(model, view) {
  const fullBox = new THREE.Box3().setFromObject(model);
  if (fullBox.isEmpty()) throw new Error("The model contains no visible geometry.");
  model.position.sub(fullBox.getCenter(new THREE.Vector3()));
  model.updateMatrixWorld(true);
  const byNode = view.by_node && typeof view.by_node === "object" ? view.by_node : {};
  const colors = view.colors && typeof view.colors === "object" ? view.colors : {};
  const originalMaterials = new Set();

  model.traverse(object => {
    if (!object.isMesh) return;
    const category = categoryFor(object, byNode);
    object.userData.samCategory = category;
    addCategoryMesh(category, object);
    const source = Array.isArray(object.material) ? object.material : [object.material];
    source.forEach(material => originalMaterials.add(material));
    if (!Object.keys(colors).length || !colors[category]) return;
    let material = state.materials.get(category);
    if (!material) {
      material = new THREE.MeshStandardMaterial({ color: colors[category], roughness: 0.9, metalness: 0, side: THREE.DoubleSide });
      state.materials.set(category, material);
    }
    object.material = material;
  });

  const referencedMaterials = new Set();
  const referencedTextures = new Set();
  model.traverse(object => {
    if (!object.isMesh) return;
    const materials = Array.isArray(object.material) ? object.material : [object.material];
    for (const material of materials) {
      referencedMaterials.add(material);
      for (const value of Object.values(material)) if (value && value.isTexture) referencedTextures.add(value);
    }
  });
  const disposedTextures = new Set();
  for (const material of originalMaterials) {
    if (referencedMaterials.has(material)) continue;
    for (const value of Object.values(material)) {
      if (value && value.isTexture && !referencedTextures.has(value) && !disposedTextures.has(value)) {
        disposedTextures.add(value);
        value.dispose();
      }
    }
    material.dispose();
  }
  state.modelRoot.add(model);
}

function setCategoryVisible(category, visible) {
  state.categoryVisible.set(category, visible);
  for (const mesh of state.categoryMeshes.get(category) || []) mesh.visible = visible;
  requestDraw();
}

function isMeshVisible(mesh) {
  for (let node = mesh; node; node = node.parent) if (!node.visible) return false;
  return true;
}

function visibleBox() {
  const box = new THREE.Box3();
  let hasMesh = false;
  for (const meshes of state.categoryMeshes.values()) {
    for (const mesh of meshes) {
      if (!isMeshVisible(mesh)) continue;
      const part = new THREE.Box3().setFromObject(mesh);
      if (!part.isEmpty()) {
        box.union(part);
        hasMesh = true;
      }
    }
  }
  return hasMesh ? box : null;
}

function sampleVisiblePoints(maxPoints = 20000) {
  const meshes = [];
  let total = 0;
  for (const list of state.categoryMeshes.values()) {
    for (const mesh of list) {
      if (!isMeshVisible(mesh) || !mesh.geometry || !mesh.geometry.attributes || !mesh.geometry.attributes.position) continue;
      meshes.push(mesh);
      total += mesh.geometry.attributes.position.count;
    }
  }
  const stride = Math.max(1, Math.ceil(total / maxPoints));
  const points = [];
  const v = new THREE.Vector3();
  for (const mesh of meshes) {
    mesh.updateWorldMatrix(true, false);
    const position = mesh.geometry.attributes.position;
    const count = position.count;
    // Per-mesh stride: every mesh contributes at least min(count, 8) real vertices.
    const meshStride = Math.max(1, Math.min(stride, Math.floor(count / 8)));
    for (let i = 0; i < count; i += meshStride) {
      v.fromBufferAttribute(position, i).applyMatrix4(mesh.matrixWorld);
      points.push(v.x, v.y, v.z);
    }
    if (count > 0 && (count - 1) % meshStride !== 0) {
      v.fromBufferAttribute(position, count - 1).applyMatrix4(mesh.matrixWorld);
      points.push(v.x, v.y, v.z);
    }
  }
  return points;
}

function fit(keepAngle = false) {
  const box = visibleBox();
  if (!box || !state.camera) return;
  const size = box.getSize(new THREE.Vector3());
  const target = box.getCenter(new THREE.Vector3());
  let direction = new THREE.Vector3(1, Math.tan(THREE.MathUtils.degToRad(35)) * Math.SQRT2, 1).normalize();
  if (keepAngle) {
    const current = state.camera.position.clone().sub(state.controls.target);
    if (current.lengthSq() > 1e-8) direction = current.normalize();
  }
  if (dom.canvas && dom.canvas.clientWidth > 0 && dom.canvas.clientHeight > 0) {
    state.camera.aspect = dom.canvas.clientWidth / dom.canvas.clientHeight;
  }
  state.camera.updateProjectionMatrix();
  const tanV = Math.tan(THREE.MathUtils.degToRad(state.camera.fov / 2));
  const tanH = tanV * (state.camera.aspect || 1);
  const fill = 0.9;
  const camUp = Math.abs(direction.y) > 0.999 ? new THREE.Vector3(0, 0, -1) : new THREE.Vector3(0, 1, 0);
  const camRight = new THREE.Vector3().crossVectors(camUp, direction).normalize();
  const camTrueUp = new THREE.Vector3().crossVectors(direction, camRight).normalize();
  const points = sampleVisiblePoints();
  const n = points.length / 3;
  let distance = 0.5;
  if (n > 0) {
    for (let iter = 0; iter < 4; iter++) {
      distance = 0.5;
      for (let i = 0; i < n; i++) {
        const x = points[i * 3] - target.x, y = points[i * 3 + 1] - target.y, z = points[i * 3 + 2] - target.z;
        const depth = x * direction.x + y * direction.y + z * direction.z;
        const px = x * camRight.x + y * camRight.y + z * camRight.z;
        const py = x * camTrueUp.x + y * camTrueUp.y + z * camTrueUp.z;
        distance = Math.max(distance, depth + Math.abs(px) / (tanH * fill), depth + Math.abs(py) / (tanV * fill));
      }
      let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
      for (let i = 0; i < n; i++) {
        const x = points[i * 3] - target.x, y = points[i * 3 + 1] - target.y, z = points[i * 3 + 2] - target.z;
        const depth = x * direction.x + y * direction.y + z * direction.z;
        const px = x * camRight.x + y * camRight.y + z * camRight.z;
        const py = x * camTrueUp.x + y * camTrueUp.y + z * camTrueUp.z;
        const range = Math.max(distance - depth, 1e-6);
        minX = Math.min(minX, px / range); maxX = Math.max(maxX, px / range);
        minY = Math.min(minY, py / range); maxY = Math.max(maxY, py / range);
      }
      target.addScaledVector(camRight, ((minX + maxX) / 2) * distance);
      target.addScaledVector(camTrueUp, ((minY + maxY) / 2) * distance);
    }
    // final distance for the recentred target
    distance = 0.5;
    for (let i = 0; i < n; i++) {
      const x = points[i * 3] - target.x, y = points[i * 3 + 1] - target.y, z = points[i * 3 + 2] - target.z;
      const depth = x * direction.x + y * direction.y + z * direction.z;
      const px = x * camRight.x + y * camRight.y + z * camRight.z;
      const py = x * camTrueUp.x + y * camTrueUp.y + z * camTrueUp.z;
      distance = Math.max(distance, depth + Math.abs(px) / (tanH * fill), depth + Math.abs(py) / (tanV * fill));
    }
  }
  state.controls.target.copy(target);
  state.camera.position.copy(target).addScaledVector(direction, distance);
  state.camera.near = Math.max(distance / 1000, 0.01);
  state.camera.far = Math.max(distance + size.length() * 4, distance * 10, 100);
  state.camera.updateProjectionMatrix();
  state.controls.update();
  requestDraw();
}

function saveInitialView() {
  state.initialView = {
    position: state.camera.position.clone(),
    target: state.controls.target.clone(),
    near: state.camera.near,
    far: state.camera.far,
  };
}

function resetView() {
  if (state.camera && visibleBox()) {
    fit(false);
    return;
  }
  if (!state.initialView) return;
  state.camera.position.copy(state.initialView.position);
  state.controls.target.copy(state.initialView.target);
  state.camera.near = state.initialView.near;
  state.camera.far = state.initialView.far;
  state.camera.updateProjectionMatrix();
  state.controls.update();
  requestDraw();
}

function buildLegend(view) {
  const order = Array.isArray(view.order) ? view.order : Array.from(state.categoryMeshes.keys());
  const labels = view.labels && typeof view.labels === "object" ? view.labels : {};
  const counts = view.counts && typeof view.counts === "object" ? view.counts : {};
  const colors = view.colors && typeof view.colors === "object" ? view.colors : {};
  dom.legend.replaceChildren();
  const actions = document.createElement("div");
  actions.className = "ifc-legend-actions";
  for (const [label, visible] of [[text("all"), true], [text("none"), false]]) {
    const button = document.createElement("button");
    button.type = "button";
    button.textContent = label;
    button.addEventListener("click", () => {
      for (const category of order) setCategoryVisible(category, visible);
      for (const input of dom.legend.querySelectorAll("input[data-category]")) input.checked = visible;
    });
    actions.append(button);
  }
  dom.legend.append(actions);
  for (const category of order) {
    const row = document.createElement("label");
    row.className = "ifc-category-row";
    const input = document.createElement("input");
    input.type = "checkbox";
    input.dataset.category = category;
    input.checked = state.categoryVisible.get(category) !== false;
    const swatch = document.createElement("i");
    swatch.className = "ifc-category-swatch";
    swatch.style.background = /^#[0-9a-f]{6}$/i.test(colors[category] || "") ? colors[category] : "#9aa5b1";
    const name = document.createElement("span");
    name.className = "ifc-category-name";
    name.textContent = labels[category] || category;
    const count = document.createElement("small");
    count.className = "ifc-category-count";
    count.textContent = formatCount(counts[category]);
    row.append(input, swatch, name, count);
    dom.legend.append(row);
  }
  dom.legend.onchange = event => {
    const input = event.target;
    if (!(input instanceof HTMLInputElement) || !input.dataset.category) return;
    setCategoryVisible(input.dataset.category, input.checked);
  };
}

async function loadLevel(level) {
  if (!level || !level.ifc) {
    setStatus(text("noIfc"));
    return;
  }
  disposeCurrentModel();
  const token = ++state.loadToken;
  state.currentLevel = level;
  setStatus(text("loadingView"));
  try {
    const view = await fetchJson(modelUrl(level.ifc, "view.json"));
    if (token !== state.loadToken) return;
    state.view = view;
    const hidden = new Set(Array.isArray(view.hidden) ? view.hidden : []);
    const loader = new GLTFLoader();
    loader.load(
      modelUrl(level.ifc, "model.glb"),
      gltf => {
        if (token !== state.loadToken) return;
        try {
          prepareModel(gltf.scene, view);
          for (const category of state.categoryMeshes.keys()) setCategoryVisible(category, !hidden.has(category));
          buildLegend(view);
          fit();
          saveInitialView();
          setStatus(text("ready"), 2000);
          requestDraw();
        } catch (error) {
          showError(error, "loadFailed");
        }
      },
      progress => {
        if (token !== state.loadToken) return;
        const percent = progress.total > 0 ? Math.min(100, Math.round((progress.loaded / progress.total) * 100)) : 0;
        setStatus(replace("loading", { percent }));
      },
      error => {
        if (token !== state.loadToken) return;
        showError(error, "loadFailed");
      },
    );
  } catch (error) {
    if (token === state.loadToken) showError(error, "loadFailed");
  }
}

function chooseLevel(project) {
  const levels = Array.isArray(project.levels) ? project.levels.filter(level => level && level.ifc) : [];
  state.levels = levels;
  const requested = new URLSearchParams(window.location.search).get("level");
  return levels.find(level => level.key === requested)
    || levels.find(level => level.key === project.default_level)
    || levels[0]
    || null;
}

function setLevelSelect(selected) {
  dom.levelSelect.replaceChildren();
  for (const level of state.levels) {
    const option = document.createElement("option");
    option.value = level.key;
    option.textContent = level.label || level.key;
    option.selected = level.key === selected.key;
    dom.levelSelect.append(option);
  }
  dom.levelControl.hidden = state.levels.length <= 1;
}

function switchLevel(key) {
  const level = state.levels.find(candidate => candidate.key === key);
  if (!level || level === state.currentLevel) return;
  const url = new URL(window.location.href);
  url.searchParams.set("level", level.key);
  window.history.replaceState(null, "", `${url.pathname}${url.search}${url.hash}`);
  if (window.SAMShare && window.SAMShare.setLevel) window.SAMShare.setLevel(level.key);
  loadLevel(level);
}

function installEvents() {
  dom.fit.addEventListener("click", () => fit(true));
  dom.reset.addEventListener("click", resetView);
  dom.categories.addEventListener("click", () => {
    const next = dom.legend.hidden;
    dom.legend.hidden = !next;
    dom.categories.classList.toggle("is-selected", next);
    dom.categories.setAttribute("aria-expanded", String(next));
  });
  dom.levelSelect.addEventListener("change", () => switchLevel(dom.levelSelect.value));
  dom.errorButton.addEventListener("click", () => {
    dom.errorPanel.hidden = !dom.errorPanel.hidden;
    if (!dom.errorPanel.hidden) updateErrorPanel();
  });
  dom.errorClose.addEventListener("click", () => { dom.errorPanel.hidden = true; });
  dom.errorCopy.addEventListener("click", copyDiagnostics);
  dom.contextReload.addEventListener("click", () => window.location.reload());
  window.addEventListener("resize", resize);
  window.addEventListener("orientationchange", resize);
  if (typeof ResizeObserver === "function" && dom.canvas) {
    new ResizeObserver(() => resize()).observe(dom.canvas);
  }
  window.addEventListener("error", event => {
    if (event.error || event.message) recordError("window.onerror", event.error || new Error(event.message));
  });
  window.addEventListener("unhandledrejection", event => recordError("unhandledrejection", event.reason || new Error("rejected")));
  for (const type of ["gesturestart", "gesturechange", "gestureend"]) {
    document.addEventListener(type, event => event.preventDefault(), { passive: false });
  }
}

async function boot() {
  installEvents();
  try {
    setupScene();
    state.project = await waitForProject();
    if (!state.project || typeof state.project !== "object") throw new Error("Invalid project.json");
    state.language = state.project.lang === "en" ? "en" : "de";
    applyTexts();
    const level = chooseLevel(state.project);
    if (!level) {
      setStatus(text("noIfc"));
      return;
    }
    setLevelSelect(level);
    loadLevel(level);
  } catch (error) {
    showError(error, "projectFailed");
  }
}

boot();
