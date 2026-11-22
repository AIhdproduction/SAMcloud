/* SAMcloud share viewer: static Potree page (RGB/class modes, class legend toggles, control point markers). */
(function () {
  "use strict";
  const byId = id => document.getElementById(id);
  const statusBox = byId("status");
  const legendBox = byId("legend");
  const markerColours = { pass: "#26b7ef", control: "#ffb020", disabled: "#6b7c8a", needs_review: "#ff5a5a",
    apriltag: "#b06cff", passpunkt: "#ff9f1a", kontrollpunkt: "#18d6e6" };
  const absolute = path => new URL(path, window.location.href).href;
  const getJson = path => fetch(absolute(path), { cache: "no-store" }).then(response => {
    if (!response.ok) throw new Error(path + ": " + response.status);
    return response.json();
  });
  const formatCount = value => Number(value || 0).toLocaleString("de-CH");

  /* Touch navigation (own controller; Potree's bundled handlers are blocked in the capture phase):
     1 finger = orbit around the pivot, 2 fingers = pinch zoom towards the pinch centre + parallel pan,
     double tap = zoom to the tapped point. Only listeners on the render area call preventDefault, so the
     toolbar and legend (siblings of the area) keep their normal tap and scroll behaviour. */
  function installTouchControls(viewer, area, onTouch) {
    const view = () => viewer.scene.view;
    const state = { span: 0, pivot: null, anchor: null, points: null, lastTap: null, moved: 0, startTime: 0 };
    const read = event => Array.from(event.touches).map(t => ({ x: t.clientX, y: t.clientY }));
    const info = points => {
      const cx = points.reduce((s, p) => s + p.x, 0) / points.length;
      const cy = points.reduce((s, p) => s + p.y, 0) / points.length;
      const dist = points.length > 1 ? Math.hypot(points[0].x - points[1].x, points[0].y - points[1].y) : 0;
      return { cx, cy, dist };
    };
    const stopMotion = () => {
      try { viewer.earthControls.stop(); viewer.earthControls.stopTweens(); } catch (_error) { /* optional */ }
      try { if (window.TWEEN) window.TWEEN.removeAll(); } catch (_error) { /* optional */ }
    };
    const pixelsToWorld = () => {
      const rect = area.getBoundingClientRect();
      const fov = (viewer.getFOV ? viewer.getFOV() : 60) * Math.PI / 180;
      return 2 * (state.span || view().radius) * Math.tan(fov / 2) / Math.max(rect.height, 1);
    };
    /* Verified call: canvas-relative pixels as THREE.Vector2, active camera, viewer, point clouds. */
    const flags = window.samcloudShareFlags = window.samcloudShareFlags || { pickingDisabled: false };
    const pickAt = (x, y) => {
      if (flags.pickingDisabled) return null;
      try {
        const canvas = viewer.renderer && viewer.renderer.domElement;
        const rect = (canvas && canvas.getBoundingClientRect ? canvas : area).getBoundingClientRect();
        const mouse = typeof THREE !== "undefined" && THREE.Vector2 ? new THREE.Vector2(x - rect.left, y - rect.top)
          : { x: x - rect.left, y: y - rect.top };
        const hit = Potree.Utils.getMousePointCloudIntersection(mouse, viewer.scene.getActiveCamera(), viewer,
          viewer.scene.pointclouds);
        return hit && hit.location ? hit.location.clone() : null;
      } catch (error) {
        /* e.g. the pick shader cannot compile on this GPU: never pick again, pivot falls back to node/bbox centres */
        state.pickError = String(error);
        flags.pickingDisabled = true;
        if (window.samcloudShareRobust && window.samcloudShareRobust.record) window.samcloudShareRobust.record("pick", error);
        return null;
      }
    };
    const vectorOf = (x, y, z) => view().position.clone().set(x, y, z);
    /* centre of the loaded octree nodes weighted by point count (follows the real data, not a loose octree cube) */
    const nodeCentre = () => {
      try {
        let sum = [0, 0, 0];
        let weight = 0;
        const info = { nodes: 0, withBox: 0, counted: 0 };
        for (const pc of viewer.scene.pointclouds) {
          pc.updateMatrixWorld && pc.updateMatrixWorld(true);
          for (const node of pc.visibleNodes || []) {
            info.nodes += 1;
            const geometry = node.geometryNode || {};
            let box = null;
            let matrix = pc.matrixWorld;
            if (node.getBoundingBox) box = node.getBoundingBox();
            box = box || node.boundingBox || geometry.boundingBox;
            if (!box && node.sceneNode && node.sceneNode.boundingBox) {  // already in the scene node's frame
              box = node.sceneNode.boundingBox;
              node.sceneNode.updateMatrixWorld && node.sceneNode.updateMatrixWorld(true);
              matrix = node.sceneNode.matrixWorld;
            }
            if (!box) continue;
            info.withBox += 1;
            let count = Number((node.getNumPoints ? node.getNumPoints() : 0) || node.numPoints || geometry.numPoints || 0);
            if (!(count > 0)) count = 1;  // size unknown (still -1 while loading): unweighted
            else info.counted += 1;
            const world = box.clone().applyMatrix4(matrix);
            sum = [sum[0] + count * (world.min.x + world.max.x) / 2, sum[1] + count * (world.min.y + world.max.y) / 2,
              sum[2] + count * (world.min.z + world.max.z) / 2];
            weight += count;
          }
        }
        state.nodeInfo = info;
        state.nodeWeight = weight;
        return weight > 0 ? vectorOf(sum[0] / weight, sum[1] / weight, sum[2] / weight) : null;
      } catch (error) { state.nodeError = String(error); return null; }
    };
    /* last resort: tight bounding box (else bounding box) of the clouds */
    const cloudCentre = () => {
      try {
        let box = null;
        const pc = viewer.scene.pointclouds[0];
        if (pc && pc.pcoGeometry && pc.pcoGeometry.tightBoundingBox) {
          box = pc.pcoGeometry.tightBoundingBox.clone().applyMatrix4(pc.matrixWorld);
        }
        box = box || viewer.getBoundingBox(viewer.scene.pointclouds);
        if (!box || !box.min || !box.max) return null;
        return vectorOf((box.min.x + box.max.x) / 2, (box.min.y + box.max.y) / 2, (box.min.z + box.max.z) / 2);
      } catch (_error) { return null; }
    };
    /* pick on a coarse grid, nearest to the screen centre first (the toolbar may hide the middle of the cloud) */
    const gridPick = () => {
      const rect = area.getBoundingClientRect();
      const cells = [];
      for (let i = 0; i < 7; i++) {
        for (let j = 0; j < 7; j++) cells.push([(i + 0.5) / 7, (j + 0.5) / 7]);
      }
      cells.sort((p, q) => Math.hypot(p[0] - 0.5, p[1] - 0.5) - Math.hypot(q[0] - 0.5, q[1] - 0.5));
      for (const [u, w] of cells) {
        const hit = pickAt(rect.left + u * rect.width, rect.top + w * rect.height);
        if (hit) return hit;
      }
      return null;
    };
    /* Pivot from the DATA: pick at the screen centre, else under the finger, else the bounding-box centre; the view
       radius is then the real distance to it (the fit may leave a meaningless radius). If the pivot is far off the
       view axis the camera turns to it so it stays on screen. */
    const determinePivot = (fingerX, fingerY) => {
      const v = view();
      const rect = area.getBoundingClientRect();
      const startPitch = v.pitch;
      const candidates = [
        () => pickAt(rect.left + rect.width / 2, rect.top + rect.height / 2),
        gridPick,
        () => (fingerX === undefined ? null : pickAt(fingerX, fingerY)),
        nodeCentre,
        cloudCentre,
      ];
      /* A pivot is only usable when it is in front of a camera that looks down: a pivot above the camera would flip
         the pitch to positive (view into empty space), so such candidates are discarded for the next fallback. */
      const usable = pivot => {
        if (!pivot || !(pivot.distanceTo(v.position) > 0.05)) return false;
        return !(startPitch < 0 && pivot.z >= v.position.z);
      };
      for (const candidate of candidates) {
        const picked = candidate();
        if (!usable(picked)) continue;
        /* The camera never turns here (a lookAt after a pan snapped the view back to the data centre): the pivot is
           the point of the data projected onto the current view axis, at the real depth of the data. */
        const depth = v.direction.clone().dot(picked.clone().sub(v.position));
        if (!(depth > 0.05)) continue;
        const pivot = v.position.clone().add(v.direction.clone().multiplyScalar(depth));
        v.radius = depth;
        return pivot;
      }
      return v.getPivot();
    };
    const livePivot = () => view().getPivot();
    const MAX_PITCH = Math.PI / 2 - 0.05;
    /* Orbit: the camera is moved back onto the sphere of ``radius`` around the pivot determined at touchstart. */
    const orbit = (dx, dy) => {
      const v = view();
      const pivot = state.pivot || livePivot();
      v.yaw -= dx * 0.006;
      v.pitch = Math.max(-MAX_PITCH, Math.min(MAX_PITCH, v.pitch - dy * 0.006));
      v.position.copy(pivot).sub(v.direction.clone().multiplyScalar(v.radius));
    };
    /* Pinch: scale the camera position (and radius) about the anchor, then pan with the centre movement. */
    const zoomAndPan = (previous, current) => {
      const v = view();
      const scale = pixelsToWorld();
      if (previous.dist > 0 && current.dist > 0) {
        const factor = Math.min(Math.max(previous.dist / current.dist, 0.5), 2);
        if (v.radius * factor > 0.05) {
          const anchor = state.anchor || livePivot();
          v.position.sub(anchor).multiplyScalar(factor).add(anchor);
          v.radius *= factor;
          state.span = (state.span || v.radius) * factor;
        }
      }
      const beforePan = v.position.clone();
      v.pan(-(current.cx - previous.cx) * scale, (current.cy - previous.cy) * scale);
      /* the pan moves the pivot and the anchor with the camera: a following one-finger orbit must start HERE */
      const moved = v.position.clone().sub(beforePan);
      if (state.anchor) state.anchor.add(moved);
      state.pivot = livePivot();
    };
    const doubleTap = (x, y) => {
      const rect = area.getBoundingClientRect();
      const mouse = { x: x - rect.left, y: y - rect.top };
      try {
        if (flags.pickingDisabled) throw new Error("picking disabled");
        const before = view().position.clone();
        viewer.earthControls.zoomToLocation(mouse);
        if (before.equals(view().position) && !viewer.earthControls.tweens.length) throw new Error("no hit");
      } catch (_error) {
        view().radius = Math.max(view().radius * 0.5, 0.05);
        const pivot = view().getPivot();
        view().position.copy(pivot).sub(view().direction.multiplyScalar(view().radius));
      }
    };
    const guard = handler => event => {
      event.preventDefault();
      event.stopPropagation();
      handler(event);
    };
    const options = { passive: false, capture: true };
    area.addEventListener("touchstart", guard(event => {
      stopMotion();
      if (onTouch) onTouch();
      state.points = read(event);
      state.residual = false;
      const centre = info(state.points);
      state.pivot = determinePivot(centre.cx, centre.cy);  // from the data, not from a possibly meaningless radius
      state.span = view().position.distanceTo(state.pivot);
      state.anchor = event.touches.length === 2 ? (pickAt(centre.cx, centre.cy) || state.pivot) : null;
      if (state.anchor) state.span = view().position.distanceTo(state.anchor);
      if (event.touches.length === 1) { state.moved = 0; state.startTime = Date.now(); }
      else state.moved = Infinity;
    }), options);
    area.addEventListener("touchmove", guard(event => {
      const current = read(event);
      const previous = state.points;
      state.points = current;
      if (state.residual) return;  // remaining finger of a two-finger gesture: no accidental orbit around a stale pivot
      if (!previous || previous.length !== current.length || !current.length) return;
      if (current.length === 1) {
        const dx = current[0].x - previous[0].x;
        const dy = current[0].y - previous[0].y;
        state.moved += Math.hypot(dx, dy);
        orbit(dx, dy);
      } else {
        state.moved = Infinity;
        zoomAndPan(info(previous), info(current));
      }
    }), options);
    const finish = guard(event => {
      const touch = event.changedTouches && event.changedTouches[0];
      const isTap = event.type === "touchend" && touch && event.touches.length === 0 && state.moved < 10
        && Date.now() - state.startTime < 350;
      state.residual = event.touches.length > 0 && (state.residual || (!!state.points && state.points.length > 1));
      state.points = event.touches.length ? read(event) : null;
      if (!isTap) return;
      const now = Date.now();
      const last = state.lastTap;
      if (last && now - last.time < 350 && Math.hypot(touch.clientX - last.x, touch.clientY - last.y) < 40) {
        state.lastTap = null;
        doubleTap(touch.clientX, touch.clientY);
      } else {
        state.lastTap = { time: now, x: touch.clientX, y: touch.clientY };
      }
    });
    area.addEventListener("touchend", finish, options);
    area.addEventListener("touchcancel", finish, options);
    return { orbit, zoomAndPan, doubleTap, state, determinePivot };
  }

  /* Mobile robustness: device profile, crash guard (Potree's crash page only after repeated failures), WebGL context
     loss handling and an error log with a details panel (the stock crash page shows an empty console on phones). */
  function installRobustness(viewer, hooks, win) {
    win = win || window;
    const profile = {};
    const canvas = viewer.renderer && viewer.renderer.domElement;
    const gl = viewer.renderer && viewer.renderer.getContext ? viewer.renderer.getContext() : null;
    const agent = (win.navigator && win.navigator.userAgent) || "";
    /* iPhone/iPad, including iPadOS which reports a desktop (Macintosh) user agent */
    profile.ios = /iPhone|iPad|iPod/.test(agent)
      || (((win.navigator && win.navigator.maxTouchPoints) || 0) > 1 && /Macintosh/.test(agent));
    profile.coarse = profile.ios || !!((win.matchMedia && win.matchMedia("(pointer: coarse)").matches)
      || ("ontouchstart" in win && Math.min(win.screen.width, win.screen.height) < 900));
    const flags = win.samcloudShareFlags = win.samcloudShareFlags || { pickingDisabled: false };
    const webgl2 = typeof win.WebGL2RenderingContext !== "undefined" && gl instanceof win.WebGL2RenderingContext;
    const edlSupported = !!gl && (webgl2 || !!gl.getExtension("WEBGL_depth_texture"));
    /* iOS kills tabs silently under GPU memory pressure: smaller budget there */
    profile.budget = profile.ios ? 700000 : (profile.coarse ? 1000000 : 3000000);
    /* Potree's pick shader (paraboloid point shape) writes gl_FragDepthEXT: WebGL1 without EXT_frag_depth cannot
       compile it. Picks then use the CIRCLE shape (only Potree.PointCloudOctree.pick reads PointShape.PARABOLOID). */
    profile.fragDepth = webgl2 || (!!gl && !!gl.getExtension("EXT_frag_depth"));
    if (!profile.fragDepth && win.Potree && win.Potree.PointShape) {
      try { win.Potree.PointShape.PARABOLOID = win.Potree.PointShape.CIRCLE; } catch (_error) { flags.pickingDisabled = true; }
    }
    profile.edl = !profile.coarse && edlSupported;
    viewer.setPointBudget(profile.budget);
    viewer.setEDLEnabled(profile.edl);
    if (profile.coarse && viewer.renderer.setPixelRatio) viewer.renderer.setPixelRatio(Math.min(win.devicePixelRatio || 1, 1.5));

    const errors = win.samcloudShareViewerErrors = [];
    const record = (kind, error) => {
      const message = error && error.message ? String(error.message) : String(error);
      errors.push({ kind, message, stack: error && error.stack ? String(error.stack) : "", time: new Date().toISOString() });
      if (errors.length > 50) errors.shift();
      if (hooks && hooks.errorRecorded) hooks.errorRecorded();
    };
    const reduce = () => {
      profile.budget = Math.max(250000, Math.floor(profile.budget / 2));
      viewer.setPointBudget(profile.budget);
      viewer.setEDLEnabled(false);
      profile.edl = false;
    };
    const originalCrash = viewer.onCrash.bind(viewer);
    const crashTimes = [];
    let shaderReduced = false;
    viewer.onCrash = error => {
      const now = Date.now();
      record("render", error);
      if (/could not compile shader|frag_?depth/i.test(String((error && error.message) || error))) {
        /* a shader problem is not a memory problem: stop picking, reduce the budget at most once, keep rendering */
        flags.pickingDisabled = true;
        if (!shaderReduced) { shaderReduced = true; reduce(); }
        if (hooks && hooks.notify) hooks.notify("Darstellungsfehler \u2013 Punktdichte reduziert");
        return;
      }
      crashTimes.push(now);
      while (crashTimes.length && now - crashTimes[0] > 10000) crashTimes.shift();
      if (crashTimes.length >= 3) { originalCrash(error); return; }  // persistent: Potree's own crash page
      reduce();
      if (hooks && hooks.notify) hooks.notify("Darstellungsfehler – Punktdichte reduziert");
    };
    /* an exception in update() would leave the animation loop unscheduled; keep it running */
    const originalUpdate = viewer.update.bind(viewer);
    viewer.update = (delta, timestamp) => {
      try { originalUpdate(delta, timestamp); } catch (error) { viewer.onCrash(error); }
    };

    if (canvas && canvas.addEventListener) {
      canvas.addEventListener("webglcontextlost", event => {
        event.preventDefault();
        record("webglcontextlost", new Error("WebGL context lost"));
        if (hooks && hooks.notify) hooks.notify("Grafikkontext verloren – wird wiederhergestellt");
      }, false);
      canvas.addEventListener("webglcontextrestored", () => {
        reduce();
        if (hooks && hooks.notify) hooks.notify("Grafikkontext wiederhergestellt – Punktdichte reduziert");
      }, false);
    }
    win.addEventListener("error", event => {
      if (event.error || event.message) record("window.onerror", event.error || new Error(event.message));
    });
    win.addEventListener("unhandledrejection", event => record("unhandledrejection", event.reason || new Error("rejected")));

    const debugInfo = () => {
      const lines = ["SAMcloud share viewer error details", ""];
      for (const item of errors) lines.push(`[${item.time}] ${item.kind}: ${item.message}`, item.stack, "");
      if (!errors.length) lines.push("(no errors recorded)", "");
      lines.push(`userAgent: ${win.navigator ? win.navigator.userAgent : "?"}`, `devicePixelRatio: ${win.devicePixelRatio}`,
        `canvas: ${canvas ? `${canvas.width}x${canvas.height} (css ${canvas.clientWidth}x${canvas.clientHeight})` : "?"}`,
        `mobile profile: ${profile.coarse}`, `iOS: ${profile.ios}`, `EXT_frag_depth/WebGL2: ${profile.fragDepth}`,
        `picking disabled: ${flags.pickingDisabled}`, `point budget: ${profile.budget}`,
        `EDL: ${viewer.getEDLEnabled ? viewer.getEDLEnabled() : profile.edl}`);
      if (gl) {
        const pick = name => { try { return gl.getParameter(gl[name]); } catch (_error) { return "?"; } };
        lines.push(`WebGL VERSION: ${pick("VERSION")}`, `SHADING_LANGUAGE_VERSION: ${pick("SHADING_LANGUAGE_VERSION")}`);
        try {
          const debugInfoExt = gl.getExtension("WEBGL_debug_renderer_info");
          if (debugInfoExt) {
            lines.push(`UNMASKED_RENDERER: ${gl.getParameter(debugInfoExt.UNMASKED_RENDERER_WEBGL)}`,
              `UNMASKED_VENDOR: ${gl.getParameter(debugInfoExt.UNMASKED_VENDOR_WEBGL)}`);
          }
        } catch (_error) { /* optional */ }
        lines.push(`MAX_TEXTURE_SIZE: ${pick("MAX_TEXTURE_SIZE")}`, `MAX_VERTEX_ATTRIBS: ${pick("MAX_VERTEX_ATTRIBS")}`);
        for (const name of ["EXT_frag_depth", "WEBGL_depth_texture", "OES_texture_float", "OES_element_index_uint"]) {
          let available = false;
          try { available = !!gl.getExtension(name); } catch (_error) { /* ignore */ }
          lines.push(`${name}: ${available ? "yes" : "no"}`);
        }
      }
      return lines.join("\n");
    };
    return { profile, errors, record, debugInfo, reduce, flags };
  }
  window.samcloudInstallRobustness = installRobustness;

  /* iOS Safari ignores user-scalable=no and fires gesture events for page pinch-zoom: cancel them. */
  function blockPageZoom(doc) {
    if (!doc || !doc.addEventListener) return;
    for (const type of ["gesturestart", "gesturechange", "gestureend"]) {
      doc.addEventListener(type, event => event.preventDefault(), { passive: false });
    }
  }
  window.samcloudBlockPageZoom = blockPageZoom;
  blockPageZoom(document);
  window.samcloudInstallTouchControls = installTouchControls;
  function start() {
    if (typeof Potree === "undefined" || typeof THREE === "undefined") { setTimeout(start, 100); return; }
    const viewer = new Potree.Viewer(byId("potree-area"));
    const errorButton = byId("error-button");
    const errorPanel = byId("error-panel");
    const errorText = byId("error-text");
    const robust = installRobustness(viewer, {
      notify: message => {
        statusBox.textContent = "";
        const text = document.createElement("span");
        text.textContent = message + " ";
        const close = document.createElement("button");
        close.type = "button";
        close.textContent = "\u00d7";
        close.setAttribute("aria-label", "Schliessen");
        close.addEventListener("click", () => { statusBox.textContent = ""; });
        statusBox.append(text, close);
      },
      errorRecorded: () => { errorButton.hidden = false; if (!errorPanel.hidden) errorText.textContent = robust.debugInfo(); },
    });
    window.samcloudShareRobust = robust;
    errorButton.addEventListener("click", () => { errorText.textContent = robust.debugInfo(); errorPanel.hidden = !errorPanel.hidden; });
    byId("error-close").addEventListener("click", () => { errorPanel.hidden = true; });
    byId("error-copy").addEventListener("click", () => {
      const text = robust.debugInfo();
      const fallback = () => {
        const area = document.createElement("textarea");
        area.value = text;
        document.body.append(area);
        area.select();
        try { document.execCommand("copy"); } catch (_error) { /* ignore */ }
        area.remove();
      };
      if (navigator.clipboard && navigator.clipboard.writeText) navigator.clipboard.writeText(text).catch(fallback);
      else fallback();
    });
    viewer.setFOV(60);
    viewer.setBackground("black");
    const earth = viewer.earthControls;
    /* left drag rotates, right drag pans (same as the SAMcloud viewer); touch input is unaffected */
    const swapButtons = handler => event => {
      const drag = event.drag;
      if (!drag) return handler(event);
      const original = drag.mouse;
      if (original === Potree.MOUSE.LEFT) drag.mouse = Potree.MOUSE.RIGHT;
      else if (original === Potree.MOUSE.RIGHT) drag.mouse = Potree.MOUSE.LEFT;
      try { return handler(event); } finally { drag.mouse = original; }
    };
    for (const type of ["drag", "drop"]) earth._listeners[type] = (earth._listeners[type] || []).map(swapButtons);
    viewer.setControls(earth);
    window.samcloudShareViewer = viewer; // for diagnostics
    window.samcloudShareTouch = installTouchControls(viewer, byId("potree-area"), () => { touched = true; });

    const scheme = { DEFAULT: { visible: true, name: "Andere", color: [0.7, 0.7, 0.7, 1.0] } };
    let cloud = null;
    let markersVisible = true;
    const markerObjects = [];

    let touched = false;  // after the first user touch no automatic fit may overwrite the view
    /* The camera must start ABOVE the data looking down at 45 degrees: set that orientation, then fit. The first
       fit can run before the octree hierarchy is loaded (camera stays at the origin), so it is repeated until the
       camera has really left the origin. Never after the first user touch. */
    const defaultOrientation = () => { const v = viewer.scene.view; v.yaw = Math.PI / 4; v.pitch = -Math.PI / 4; };
    const established = () => {
      const v = viewer.scene.view;
      return v.position.length() > 1e-6 && v.pitch < 0;
    };
    const fit = force => {
      if (touched && force !== true) return;
      if (force === true || !established()) defaultOrientation();
      try { viewer.fitToScreen(0.55); } catch (_error) { /* not loaded yet */ }
    };
    /* The initial view is established exactly like the reset button, directly after the point cloud was added: its
       bounding box is known from metadata.json at once. (Waiting for visible octree nodes would deadlock: with the
       camera at the origin nothing is visible.) Repeated shortly after and never once the user has touched. */
    const cloudReady = () => {
      const clouds = viewer.scene.pointclouds;
      return clouds.length > 0 && !!clouds[0].boundingBox && !clouds[0].boundingBox.isEmpty();
    };
    const initialFit = () => {
      fit(true);
      for (const delay of [500, 1500]) setTimeout(() => { if (!touched) fit(true); }, delay);
    };
    const refitIfUntouched = () => { if (!touched && cloudReady()) fit(true); };
    window.addEventListener("resize", refitIfUntouched);
    window.addEventListener("orientationchange", () => setTimeout(refitIfUntouched, 300));
    getJson("data/legend.json").then(legend => {
      byId("project-name").textContent = legend.project_name || "Punktwolke";
      const parts = [legend.created_at, formatCount(legend.points) + " Punkte"];
      byId("project-meta").textContent = parts.filter(Boolean).join(" · ");
      document.title = (legend.project_name || "Punktwolke") + " – SAMcloud";
      buildLegend(legend.classes || []);
    }).catch(error => { statusBox.textContent = "Legende nicht verfügbar: " + error.message; });

    function buildLegend(classes) {
      const actions = document.createElement("div");
      actions.className = "share-legend-actions";
      for (const [label, value] of [["Alle", true], ["Keine", false]]) {
        const button = document.createElement("button");
        button.type = "button";
        button.textContent = label;
        button.addEventListener("click", () => {
          for (const input of legendBox.querySelectorAll("input[data-code]")) {
            input.checked = value;
            scheme[input.dataset.code].visible = value;
          }
          viewer.setClassifications(scheme);
        });
        actions.append(button);
      }
      legendBox.append(actions);
      for (const item of classes) {
        const hex = /^#[0-9A-Fa-f]{6}$/.test(item.color) ? item.color : "#b0b0b0";
        const rgb = [1, 3, 5].map(i => parseInt(hex.slice(i, i + 2), 16) / 255);
        scheme[item.code] = { visible: true, name: item.name, color: [rgb[0], rgb[1], rgb[2], 1.0] };
        const label = document.createElement("label");
        const input = document.createElement("input");
        input.type = "checkbox";
        input.checked = true;
        input.dataset.code = String(item.code);
        const swatch = document.createElement("i");
        swatch.style.background = hex;
        const name = document.createElement("span");
        name.textContent = item.name;
        const count = document.createElement("small");
        count.textContent = formatCount(item.count);
        label.append(input, swatch, name, count);
        legendBox.append(label);
      }
      viewer.setClassifications(scheme);
      legendBox.addEventListener("change", event => {
        const code = event.target.dataset && event.target.dataset.code;
        if (code === undefined) return;
        scheme[code].visible = event.target.checked;
        viewer.setClassifications(scheme);
      });
    }

    Potree.loadPointCloud(absolute("pointcloud/metadata.json"), "cloud").then(event => {
      cloud = event.pointcloud;
      viewer.scene.addPointCloud(cloud);
      cloud.material.size = 1.5;
      cloud.material.pointSizeType = Potree.PointSizeType.ADAPTIVE;
      cloud.material.shape = Potree.PointShape.CIRCLE;
      cloud.material.activeAttributeName = "rgba";
      initialFit();
      statusBox.textContent = "";
      loadMarkers();
    }).catch(error => {
      statusBox.textContent = "Punktwolke konnte nicht geladen werden (" + (error && error.message ? error.message : error)
        + "). Die Seite muss über http(s) geöffnet werden.";
    });

    function loadMarkers() {
      getJson("data/markers.json").then(payload => {
        for (const marker of payload.markers || []) {
          if (!Array.isArray(marker.coordinates) || marker.coordinates.length !== 3) continue;
          const colour = markerColours[marker.state_color] || markerColours[marker.kind] || markerColours.passpunkt;
          const isTag = marker.kind === "apriltag";
          const annotation = new Potree.Annotation({ position: new THREE.Vector3(...marker.coordinates), title: marker.name });
          viewer.scene.annotations.add(annotation);
          const sphere = new THREE.Mesh(new THREE.SphereGeometry(isTag ? 0.05 : 0.18, 20, 14),
            new THREE.MeshBasicMaterial({ color: colour, depthTest: false, transparent: true, opacity: 0.95 }));
          sphere.position.set(...marker.coordinates);
          sphere.renderOrder = 999;
          viewer.scene.scene.add(sphere);
          let square = null;
          if (Array.isArray(marker.corners) && marker.corners.length === 4) {
            const geometry = new THREE.BufferGeometry();
            geometry.setAttribute("position", new THREE.Float32BufferAttribute(marker.corners.flat(), 3));
            square = new THREE.LineLoop(geometry, new THREE.LineBasicMaterial({ color: colour, depthTest: false, transparent: true, opacity: 0.95 }));
            square.renderOrder = 998;
            viewer.scene.scene.add(square);
          }
          const element = annotation.domElement && annotation.domElement[0];
          if (element) {
            const titlebar = element.querySelector(".annotation-titlebar");
            if (titlebar) titlebar.style.background = colour;
          }
          markerObjects.push({ annotation, sphere, square });
        }
        if (!markerObjects.length) {
          const button = byId("markers-button");
          button.disabled = true;
          button.style.opacity = ".5";
          button.title = "Keine Passpunkte vorhanden";
        }
      }).catch(() => { /* markers are optional */ });
    }

    const select = (names, active) => names.forEach(name => {
      const button = document.querySelector(`[data-action="${name}"]`);
      if (button) button.classList.toggle("is-selected", name === active);
    });
    byId("toolbar").addEventListener("click", event => {
      const action = event.target.dataset && event.target.dataset.action;
      if (!action) return;
      if ((action === "rgb" || action === "classification") && cloud) {
        cloud.material.activeAttributeName = action === "rgb" ? "rgba" : "classification";
        select(["rgb", "classification"], action);
      }
      if (action === "legend") legendBox.classList.toggle("is-open");
      if (action === "reset") { touched = false; fit(true); touched = true; }
      if (action === "markers" && !event.target.disabled) {
        markersVisible = !markersVisible;
        for (const item of markerObjects) {
          item.sphere.visible = markersVisible;
          if (item.square) item.square.visible = markersVisible;
          const element = item.annotation.domElement && item.annotation.domElement[0];
          if (element) element.classList.toggle("is-hidden", !markersVisible);
        }
        event.target.classList.toggle("is-selected", markersVisible);
      }
    });
    byId("size-slider").addEventListener("input", event => { if (cloud) cloud.material.size = Number(event.target.value); });
  }
  start();
})();
