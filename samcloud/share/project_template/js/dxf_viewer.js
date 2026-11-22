/* SAMcloud 2D DXF Viewer: pure HTML5 Canvas, zero external libraries.
   Supports Android Chrome & iPhone Safari touch, pinch-zoom, pan, layers, level chooser. */
(function () {
  "use strict";

  const byId = function (id) { return document.getElementById(id); };

  const stage = byId("dxf-stage");
  const canvas = byId("dxf-canvas");
  const ctx = canvas.getContext("2d");
  const toolbar = byId("dxf-toolbar");
  const btnFit = byId("btn-fit");
  const btnLayers = byId("btn-layers");
  const levelSelect = byId("level-select");
  const legend = byId("dxf-legend");
  const layerList = byId("dxf-layer-list");
  const btnLayersAll = byId("btn-layers-all");
  const btnLayersNone = byId("btn-layers-none");
  const btnLegendClose = byId("btn-legend-close");
  const scaleBox = byId("dxf-scale");
  const scaleBar = byId("dxf-scale-bar");
  const scaleLabel = byId("dxf-scale-label");
  const statusBox = byId("dxf-status");
  const errorButton = byId("error-button");
  const errorPanel = byId("error-panel");
  const errorText = byId("error-text");
  const errorCopy = byId("error-copy");
  const errorClose = byId("error-close");

  const i18n = {
    de: {
      fit: "Einpassen",
      layers: "Ebenen",
      all: "Alle",
      none: "Keine",
      loading: "Plan wird geladen …",
      loading_progress: "Plan wird geladen ({p}%) …",
      error_loading: "Plan konnte nicht geladen werden ({error}).",
      error_missing: "Kein DXF-Plan für diese Ebene verfügbar.",
      copy: "Kopieren",
      close: "Schliessen",
      copied: "Kopiert!"
    },
    en: {
      fit: "Fit",
      layers: "Layers",
      all: "All",
      none: "None",
      loading: "Loading plan …",
      loading_progress: "Loading plan ({p}%) …",
      error_loading: "Could not load plan ({error}).",
      error_missing: "No DXF plan available for this level.",
      copy: "Copy",
      close: "Close",
      copied: "Copied!"
    }
  };

  let lang = "de";
  function t(key) {
    const table = i18n[lang] || i18n.de;
    return table[key] || (i18n.de && i18n.de[key]) || key;
  }

  // Error recording and details panel
  const recordedErrors = [];
  function recordError(kind, err) {
    const message = (err && err.message) ? String(err.message) : String(err);
    const stack = (err && err.stack) ? String(err.stack) : "";
    recordedErrors.push({ kind: kind, message: message, stack: stack, time: new Date().toISOString() });
    if (recordedErrors.length > 50) recordedErrors.shift();
    if (errorButton) errorButton.hidden = false;
    if (errorPanel && !errorPanel.hidden) errorText.textContent = getDebugInfo();
  }

  function getDebugInfo() {
    const lines = ["SAMcloud DXF viewer error details", ""];
    for (const item of recordedErrors) {
      lines.push("[" + item.time + "] " + item.kind + ": " + item.message, item.stack, "");
    }
    if (!recordedErrors.length) lines.push("(no errors recorded)", "");
    lines.push(
      "userAgent: " + navigator.userAgent,
      "devicePixelRatio: " + window.devicePixelRatio,
      "canvas: " + (canvas ? canvas.width + "x" + canvas.height + " (css " + canvas.clientWidth + "x" + canvas.clientHeight + ")" : "?"),
      "view: " + JSON.stringify(view)
    );
    return lines.join("\n");
  }

  window.addEventListener("error", function (ev) {
    recordError("window.onerror", ev.error || new Error(ev.message));
  });
  window.addEventListener("unhandledrejection", function (ev) {
    recordError("unhandledrejection", ev.reason || new Error("rejected"));
  });

  if (errorButton) {
    errorButton.addEventListener("click", function () {
      errorText.textContent = getDebugInfo();
      errorPanel.hidden = !errorPanel.hidden;
    });
  }
  if (errorClose) {
    errorClose.addEventListener("click", function () {
      errorPanel.hidden = true;
    });
  }
  if (errorCopy) {
    errorCopy.addEventListener("click", function () {
      const text = getDebugInfo();
      const fallback = function () {
        const area = document.createElement("textarea");
        area.value = text;
        document.body.appendChild(area);
        area.select();
        try { document.execCommand("copy"); } catch (_e) {}
        area.remove();
        errorCopy.textContent = t("copied");
        setTimeout(function () { errorCopy.textContent = t("copy"); }, 1500);
      };
      if (navigator.clipboard && navigator.clipboard.writeText) {
        navigator.clipboard.writeText(text).then(function () {
          errorCopy.textContent = t("copied");
          setTimeout(function () { errorCopy.textContent = t("copy"); }, 1500);
        }).catch(fallback);
      } else {
        fallback();
      }
    });
  }

  // Mobile gesture prevention on iOS
  for (const type of ["gesturestart", "gesturechange", "gestureend"]) {
    document.addEventListener(type, function (e) { e.preventDefault(); }, { passive: false });
  }

  // Device profile and DPR
  const isCoarse = /iPhone|iPad|iPod/.test(navigator.userAgent) ||
    (("maxTouchPoints" in navigator) && navigator.maxTouchPoints > 1) ||
    (window.matchMedia && window.matchMedia("(pointer: coarse)").matches);
  const dpr = Math.min(window.devicePixelRatio || 1, isCoarse ? 1.5 : 2.0);

  // View state
  const view = {
    ox: 0,
    oy: 0,
    k: 1,
    fitted: false,
    userMoved: false
  };

  let minK = 1e-6;
  let maxK = 1e6;
  let cssW = 0;
  let cssH = 0;

  // Active DXF data
  let currentDxfData = null;
  let layers = [];
  let frameRequested = false;

  function toWorld(sx, sy) {
    return {
      x: (sx - view.ox) / view.k,
      y: (view.oy - sy) / view.k
    };
  }

  function toScreen(wx, wy) {
    return {
      x: view.ox + wx * view.k,
      y: view.oy - wy * view.k
    };
  }

  function requestDraw() {
    if (frameRequested) return;
    frameRequested = true;
    window.requestAnimationFrame(function () {
      frameRequested = false;
      draw();
    });
  }

  function draw() {
    if (!canvas || !ctx) return;

    // Canvas background
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.fillStyle = (currentDxfData && currentDxfData.background) || "#0d1b2a";
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    if (!currentDxfData || !layers.length) return;

    // Stroke polylines with CAD Y-up transform
    ctx.setTransform(dpr * view.k, 0, 0, -dpr * view.k, dpr * view.ox, dpr * view.oy);
    ctx.lineWidth = 1.25 / view.k;
    ctx.lineCap = "round";
    ctx.lineJoin = "round";

    for (let i = 0; i < layers.length; i++) {
      const layer = layers[i];
      if (!layer.visible || !layer.path) continue;
      ctx.strokeStyle = layer.color;
      ctx.stroke(layer.path);
    }

    // Draw texts in screen coordinate system
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    let drawnTexts = 0;
    ctx.textAlign = "left";
    ctx.textBaseline = "middle";

    for (let i = 0; i < layers.length; i++) {
      const layer = layers[i];
      if (!layer.visible || !layer.texts || !layer.texts.length) continue;
      ctx.fillStyle = layer.color;

      for (let j = 0; j < layer.texts.length; j++) {
        const item = layer.texts[j];
        const sx = view.ox + item.x * view.k;
        const sy = view.oy - item.y * view.k;

        // Viewport culling
        if (sx < -80 || sy < -80 || sx > cssW + 80 || sy > cssH + 80) continue;

        const screenH = item.h * view.k;
        if (screenH < 7 || screenH > 400) continue;

        drawnTexts++;
        if (drawnTexts > 3000) break;

        ctx.save();
        ctx.translate(sx, sy);
        const rot = (Number(item.rot || 0) * Math.PI) / 180;
        if (rot) ctx.rotate(-rot);
        ctx.font = Math.round(screenH) + 'px system-ui, -apple-system, "Segoe UI", Roboto, sans-serif';
        ctx.fillText(item.s || item.text || "", 0, 0);
        ctx.restore();
      }

      if (drawnTexts > 3000) break;
    }

    updateScaleBar();
  }

  function updateScaleBar() {
    if (!scaleBox || !scaleBar || !scaleLabel) return;
    if (!currentDxfData || !view.k || view.k <= 0) {
      scaleBox.hidden = true;
      return;
    }

    // Units: integer coords * scale = metres
    const unitScale = currentDxfData.scale || 0.001;
    const pxPerMeter = (1 / unitScale) * view.k;
    if (!pxPerMeter || pxPerMeter <= 0) {
      scaleBox.hidden = true;
      return;
    }

    const targetMeters = 100 / pxPerMeter;
    const exp = Math.floor(Math.log10(targetMeters));
    const frac = targetMeters / Math.pow(10, exp);

    let mult = 1;
    if (frac >= 7.5) mult = 10;
    else if (frac >= 3.5) mult = 5;
    else if (frac >= 1.5) mult = 2;

    const meters = mult * Math.pow(10, exp);
    const barPx = Math.round(meters * pxPerMeter);

    let label = "";
    if (meters >= 1000) {
      label = (meters / 1000) + " km";
    } else if (meters >= 1) {
      label = meters + " m";
    } else {
      label = Math.round(meters * 100) + " cm";
    }

    scaleBar.style.width = Math.max(12, barPx) + "px";
    scaleLabel.textContent = label;
    scaleBox.hidden = false;
  }

  function resizeCanvas() {
    if (!stage || !canvas) return;
    const rect = stage.getBoundingClientRect();
    const oldW = cssW;
    const oldH = cssH;
    cssW = rect.width;
    cssH = rect.height;

    canvas.width = Math.max(1, Math.round(cssW * dpr));
    canvas.height = Math.max(1, Math.round(cssH * dpr));

    if (!view.fitted || !view.userMoved) {
      fit(true);
    } else if (oldW && oldH) {
      view.ox += (cssW - oldW) / 2;
      view.oy += (cssH - oldH) / 2;
    }

    requestDraw();
  }

  function fit(force) {
    if (!currentDxfData || !cssW || !cssH) return;
    if (view.userMoved && force !== true) return;

    const bounds = currentDxfData.bounds || [0, 0, 1000, 1000];
    const spanX = Math.max(1, bounds[2] - bounds[0]);
    const spanY = Math.max(1, bounds[3] - bounds[1]);

    // Fit with 5% margin on each edge
    const k = 0.90 * Math.min(cssW / spanX, cssH / spanY);
    view.k = k;

    minK = k * 0.05;
    maxK = k * 1000;

    const midX = (bounds[0] + bounds[2]) / 2;
    const midY = (bounds[1] + bounds[3]) / 2;

    view.ox = cssW / 2 - midX * k;
    view.oy = cssH / 2 + midY * k;
    view.fitted = true;
    view.userMoved = false;

    requestDraw();
  }

  function zoomAroundPoint(clientX, clientY, factor) {
    const rect = canvas.getBoundingClientRect();
    const sx = clientX - rect.left;
    const sy = clientY - rect.top;
    const world = toWorld(sx, sy);

    const newK = Math.min(maxK, Math.max(minK, view.k * factor));
    view.k = newK;
    view.ox = sx - world.x * newK;
    view.oy = sy + world.y * newK;
    view.userMoved = true;
    requestDraw();
  }

  // Pointer and Touch gestures
  const activePointers = new Map();
  let dragStart = null;
  let pinchState = null;
  let lastTap = null;

  canvas.addEventListener("pointerdown", function (e) {
    if (e.button !== 0 && e.pointerType === "mouse") return;
    try { canvas.setPointerCapture(e.pointerId); } catch (_err) {}
    activePointers.set(e.pointerId, { x: e.clientX, y: e.clientY });

    if (activePointers.size === 1) {
      const now = Date.now();
      if (e.pointerType === "mouse") {
        lastTap = null;
      } else if (lastTap && (now - lastTap.time < 320) && Math.hypot(e.clientX - lastTap.x, e.clientY - lastTap.y) < 25) {
        zoomAroundPoint(e.clientX, e.clientY, 2.0);
        lastTap = null;
      } else {
        lastTap = { time: now, x: e.clientX, y: e.clientY };
      }
      dragStart = { x: e.clientX, y: e.clientY, ox: view.ox, oy: view.oy };
    } else if (activePointers.size === 2) {
      const pts = Array.from(activePointers.values());
      const midX = (pts[0].x + pts[1].x) / 2;
      const midY = (pts[0].y + pts[1].y) / 2;
      const rect = canvas.getBoundingClientRect();
      const screenMidX = midX - rect.left;
      const screenMidY = midY - rect.top;

      pinchState = {
        dist: Math.hypot(pts[1].x - pts[0].x, pts[1].y - pts[0].y),
        k: view.k,
        worldMid: toWorld(screenMidX, screenMidY),
        screenMidX: screenMidX,
        screenMidY: screenMidY
      };
      dragStart = null;
    }
  });

  canvas.addEventListener("pointermove", function (e) {
    if (!activePointers.has(e.pointerId)) return;
    activePointers.set(e.pointerId, { x: e.clientX, y: e.clientY });

    if (activePointers.size === 1 && dragStart) {
      const dx = e.clientX - dragStart.x;
      const dy = e.clientY - dragStart.y;
      if (Math.hypot(dx, dy) > 2) {
        view.userMoved = true;
      }
      view.ox = dragStart.ox + dx;
      view.oy = dragStart.oy + dy;
      requestDraw();
    } else if (activePointers.size === 2 && pinchState) {
      view.userMoved = true;
      const pts = Array.from(activePointers.values());
      const dist = Math.hypot(pts[1].x - pts[0].x, pts[1].y - pts[0].y);
      const midX = (pts[0].x + pts[1].x) / 2;
      const midY = (pts[0].y + pts[1].y) / 2;
      const rect = canvas.getBoundingClientRect();
      const screenMidX = midX - rect.left;
      const screenMidY = midY - rect.top;

      const factor = dist / Math.max(1, pinchState.dist);
      const newK = Math.min(maxK, Math.max(minK, pinchState.k * factor));

      view.k = newK;
      view.ox = screenMidX - pinchState.worldMid.x * newK;
      view.oy = screenMidY + pinchState.worldMid.y * newK;
      requestDraw();
    }
  });

  function onPointerEnd(e) {
    if (activePointers.has(e.pointerId)) {
      try { canvas.releasePointerCapture(e.pointerId); } catch (_err) {}
      activePointers.delete(e.pointerId);
    }

    if (activePointers.size === 1) {
      const rem = Array.from(activePointers.values())[0];
      dragStart = { x: rem.x, y: rem.y, ox: view.ox, oy: view.oy };
      pinchState = null;
    } else if (activePointers.size === 0) {
      dragStart = null;
      pinchState = null;
    }
  }

  canvas.addEventListener("pointerup", onPointerEnd);
  canvas.addEventListener("pointercancel", onPointerEnd);

  canvas.addEventListener("wheel", function (e) {
    e.preventDefault();
    const factor = Math.exp(-e.deltaY * 0.0015);
    zoomAroundPoint(e.clientX, e.clientY, factor);
  }, { passive: false });

  canvas.addEventListener("dblclick", function (e) {
    e.preventDefault();
    zoomAroundPoint(e.clientX, e.clientY, 2.0);
  });

  // Toolbar & Legend controls
  btnFit.addEventListener("click", function () {
    view.userMoved = false;
    fit(true);
  });

  btnLayers.addEventListener("click", function () {
    legend.hidden = !legend.hidden;
    btnLayers.classList.toggle("is-selected", !legend.hidden);
  });

  btnLegendClose.addEventListener("click", function () {
    legend.hidden = true;
    btnLayers.classList.remove("is-selected");
  });

  btnLayersAll.addEventListener("click", function () {
    for (const l of layers) l.visible = true;
    for (const cb of layerList.querySelectorAll("input[type=checkbox]")) cb.checked = true;
    requestDraw();
  });

  btnLayersNone.addEventListener("click", function () {
    for (const l of layers) l.visible = false;
    for (const cb of layerList.querySelectorAll("input[type=checkbox]")) cb.checked = false;
    requestDraw();
  });

  // Build layer panel rows
  function buildLayerList() {
    layerList.innerHTML = "";
    layers.forEach(function (layer) {
      const row = document.createElement("label");
      row.className = "dxf-layer-row";

      const cb = document.createElement("input");
      cb.type = "checkbox";
      cb.checked = layer.visible;
      cb.addEventListener("change", function () {
        layer.visible = cb.checked;
        requestDraw();
      });

      const swatch = document.createElement("span");
      swatch.className = "dxf-layer-swatch";
      swatch.style.background = layer.color;

      const name = document.createElement("span");
      name.className = "dxf-layer-name";
      name.textContent = layer.name;
      name.title = layer.name;

      row.appendChild(cb);
      row.appendChild(swatch);
      row.appendChild(name);
      layerList.appendChild(row);
    });
  }

  // Load DXF data and construct Path2D in chunks
  let loadToken = 0;
  async function loadDxf(dxfUrl) {
    const token = ++loadToken;
    statusBox.hidden = false;
    statusBox.textContent = t("loading");
    currentDxfData = null;
    layers = [];
    view.fitted = false;
    view.userMoved = false;

    try {
      const res = await fetch(dxfUrl, { cache: "no-store" });
      if (!res.ok) throw new Error("HTTP " + res.status);
      const data = await res.json();
      if (token !== loadToken) return;
      currentDxfData = data;

      const rawLayers = Array.isArray(data.layers) ? data.layers : [];
      let totalPolylines = 0;
      for (const rl of rawLayers) {
        if (Array.isArray(rl.polylines)) totalPolylines += rl.polylines.length;
      }

      let processedPolylines = 0;
      let chunkCount = 0;
      const CHUNK_SIZE = 20000;

      const parsedLayers = [];

      for (let l = 0; l < rawLayers.length; l++) {
        const rl = rawLayers[l];
        const path = new Path2D();
        const polys = Array.isArray(rl.polylines) ? rl.polylines : [];

        for (let p = 0; p < polys.length; p++) {
          const poly = polys[p];
          if (!poly || poly.length < 4) continue;

          if (typeof poly[0] === "number") {
            path.moveTo(poly[0], poly[1]);
            for (let i = 2; i < poly.length; i += 2) {
              path.lineTo(poly[i], poly[i + 1]);
            }
          } else if (Array.isArray(poly[0])) {
            path.moveTo(poly[0][0], poly[0][1]);
            for (let i = 1; i < poly.length; i++) {
              path.lineTo(poly[i][0], poly[i][1]);
            }
          }

          chunkCount++;
          processedPolylines++;

          if (chunkCount >= CHUNK_SIZE) {
            chunkCount = 0;
            if (totalPolylines > 0) {
              const pct = Math.round((processedPolylines / totalPolylines) * 100);
              statusBox.textContent = t("loading_progress").replace("{p}", pct);
            }
            await new Promise(function (resolve) { setTimeout(resolve, 0); });
            if (token !== loadToken) return;
          }
        }

        parsedLayers.push({
          name: rl.name || "Layer " + (l + 1),
          color: rl.color || "#ffffff",
          visible: rl.visible !== false,
          path: path,
          texts: Array.isArray(rl.texts) ? rl.texts : []
        });
      }

      if (token !== loadToken) return;
      layers = parsedLayers;
      buildLayerList();

      statusBox.textContent = "";
      statusBox.hidden = true;

      fit(true);
      requestDraw();
    } catch (err) {
      if (token !== loadToken) return;
      recordError("loadDxf", err);
      statusBox.hidden = false;
      statusBox.textContent = t("error_loading").replace("{error}", err.message || err);
    }
  }

  // Init and project linking
  let projectInitialized = false;
  function initProject(project) {
    if (projectInitialized) return;
    projectInitialized = true;
    if (project.lang) {
      lang = project.lang === "en" ? "en" : "de";
      document.documentElement.lang = lang;
    }

    btnFit.textContent = t("fit");
    btnLayers.textContent = t("layers");
    btnLayersAll.textContent = t("all");
    btnLayersNone.textContent = t("none");

    const levels = (project && Array.isArray(project.levels)) ? project.levels : [];
    const dxfLevels = levels.filter(function (lvl) { return !!lvl.dxf; });

    let currentLevel = new URLSearchParams(window.location.search).get("level");
    if (!currentLevel && window.SAMShare && window.SAMShare.level) {
      currentLevel = window.SAMShare.level;
    }
    if (!currentLevel && project && project.default_level) {
      currentLevel = project.default_level;
    }
    if (!currentLevel && dxfLevels[0]) {
      currentLevel = dxfLevels[0].key;
    }

    // Populate level select dropdown
    if (dxfLevels.length > 1) {
      levelSelect.innerHTML = "";
      dxfLevels.forEach(function (lvl) {
        const opt = document.createElement("option");
        opt.value = lvl.key;
        opt.textContent = lvl.label || lvl.key;
        if (lvl.key === currentLevel) opt.selected = true;
        levelSelect.appendChild(opt);
      });
      levelSelect.hidden = false;

      levelSelect.addEventListener("change", function (e) {
        const newLvl = e.target.value;
        currentLevel = newLvl;
        if (window.SAMShare && window.SAMShare.setLevel) {
          window.SAMShare.setLevel(newLvl);
        }
        const url = new URL(window.location.href);
        url.searchParams.set("level", newLvl);
        window.history.replaceState(null, "", url.toString());

        const target = dxfLevels.find(function (l) { return l.key === newLvl; });
        if (target && target.dxf) {
          const root = (window.SAMShare && window.SAMShare.root) || "";
          loadDxf(root + target.dxf);
        } else {
          statusBox.hidden = false;
          statusBox.textContent = t("error_missing");
        }
      });
    } else {
      levelSelect.hidden = true;
    }

    // Find level to load
    let targetLevel = dxfLevels.find(function (l) { return l.key === currentLevel; }) || dxfLevels[0];
    if (targetLevel && targetLevel.dxf) {
      const root = (window.SAMShare && window.SAMShare.root) || "";
      loadDxf(root + targetLevel.dxf);
    } else {
      statusBox.hidden = false;
      statusBox.textContent = t("error_missing");
    }
  }

  window.addEventListener("samshare:project", function (ev) {
    initProject(ev.detail || {});
  });

  if (window.SAMShare && window.SAMShare.project) {
    initProject(window.SAMShare.project);
  }

  // Resize handling
  window.addEventListener("resize", resizeCanvas);
  if (typeof ResizeObserver === "function" && stage) {
    new ResizeObserver(function () { resizeCanvas(); }).observe(stage);
  }
  window.addEventListener("orientationchange", function () {
    setTimeout(resizeCanvas, 150);
  });

  resizeCanvas();
})();
