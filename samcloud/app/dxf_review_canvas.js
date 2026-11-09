// DXF review canvas for SAMcloud: background image + DXF vectors in one world transform.
// Culling: only entities registered in payload grid cells that intersect the viewport are drawn,
// batched per layer into one Path2D. Opacities use globalAlpha like samcloud.dxf.overlay.composite.
export default {
  template: `
    <div style="position:relative;width:100%;height:100%;min-height:420px;overflow:hidden;border:1px solid #28475b;border-radius:9px">
      <canvas ref="canvas" tabindex="0" style="display:block;width:100%;height:100%;touch-action:none;outline:none"
        @pointerdown="onDown" @pointermove="onMove" @pointerup="onUp" @pointerleave="onLeave" @wheel.prevent="onWheel"></canvas>
      <div ref="status" style="position:absolute;left:10px;bottom:8px;padding:4px 8px;border-radius:5px;background:rgba(9,19,29,.86);color:#b9ccda;font-size:12px;pointer-events:none"></div>
      <div ref="hint" style="position:absolute;right:10px;bottom:8px;padding:4px 8px;border-radius:5px;background:rgba(9,19,29,.86);color:#7f97ab;font-size:11px;pointer-events:none">Wheel: zoom · drag: pan · hold H: hide DXF</div>
      <div v-if="message" style="position:absolute;inset:0;display:grid;place-items:center;color:#96aabe;font-size:14px;pointer-events:none">{{ message }}</div>
    </div>
  `,
  props: {
    payloadUrl: String,
    imageUrl: String,
    imageBounds: Array,
    imageOpacity: Number,
    dxfOpacity: Number,
    hiddenLayers: Array,
    tool: String,
    background: Array,
  },
  data() {
    return { message: "" };
  },
  mounted() {
    this.view = { k: 1, ox: 0, oy: 0, fitted: false };
    this.layers = [];
    this.hideDxf = false;
    this.drag = null;
    this.measurePoints = [];
    this.frameRequested = false;
    this.stampFrame = 0;
    this.img = new Image();
    this.img.onload = () => this.requestDraw();
    this.observer = new ResizeObserver(() => this.resize());
    this.observer.observe(this.$el);
    this.keyDown = (event) => {
      if ((event.key === "h" || event.key === "H") && !this.isTyping(event) && !this.hideDxf) {
        this.hideDxf = true;
        this.requestDraw();
      }
    };
    this.keyUp = (event) => {
      if ((event.key === "h" || event.key === "H") && this.hideDxf) {
        this.hideDxf = false;
        this.requestDraw();
      }
    };
    this.blur = () => {
      if (this.hideDxf) {
        this.hideDxf = false;
        this.requestDraw();
      }
    };
    window.addEventListener("keydown", this.keyDown);
    window.addEventListener("keyup", this.keyUp);
    window.addEventListener("blur", this.blur);
    this.resize();
    this.loadImage();
    this.loadPayload();
  },
  unmounted() {
    if (this.observer) this.observer.disconnect();
    window.removeEventListener("keydown", this.keyDown);
    window.removeEventListener("keyup", this.keyUp);
    window.removeEventListener("blur", this.blur);
  },
  watch: {
    payloadUrl() {
      this.view.fitted = false;
      this.measurePoints = [];
      this.loadPayload();
    },
    imageUrl() {
      this.loadImage();
    },
    imageBounds() {
      this.requestDraw();
    },
    imageOpacity() {
      this.requestDraw();
    },
    dxfOpacity() {
      this.requestDraw();
    },
    hiddenLayers() {
      this.requestDraw();
    },
    background() {
      this.requestDraw();
    },
    tool() {
      this.measurePoints = [];
      this.$refs.canvas.style.cursor = this.tool === "measure" ? "crosshair" : "grab";
      this.requestDraw();
    },
  },
  methods: {
    isTyping(event) {
      const target = event.target;
      if (!target) return false;
      const tag = (target.tagName || "").toLowerCase();
      return tag === "input" || tag === "textarea" || target.isContentEditable;
    },
    absoluteUrl(url) {
      return (url.startsWith("/") ? window.path_prefix || "" : "") + url;
    },
    loadImage() {
      if (!this.imageUrl) {
        this.img.removeAttribute("src");
        this.requestDraw();
        return;
      }
      this.img.src = this.absoluteUrl(this.imageUrl);
    },
    async loadPayload() {
      this.layers = [];
      if (!this.payloadUrl) {
        this.message = "No cached DXF preview.";
        this.requestDraw();
        return;
      }
      this.message = "Loading cached DXF preview...";
      const url = this.payloadUrl;
      try {
        const response = await fetch(this.absoluteUrl(url), { cache: "no-store" });
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        const payload = await response.json();
        if (url !== this.payloadUrl) return;
        this.prepare(payload);
        this.message = "";
        let entities = 0;
        for (const layer of this.layers) entities += layer.entities.length;
        this.$emit("loaded", { entities, layers: this.layers.length });
      } catch (error) {
        this.message = `Could not load the cached DXF preview (${error.message}).`;
      }
      if (!this.view.fitted) this.fit();
      this.requestDraw();
    },
    prepare(payload) {
      this.backgroundBounds = payload.background_bounds;
      this.layers = Object.entries(payload.layers || {}).map(([name, layer]) => ({
        name,
        color: `rgb(${layer.color[0]},${layer.color[1]},${layer.color[2]})`,
        entities: layer.entities || [],
        bounds: layer.bounds,
        cellSize: layer.cell_size,
        grid: layer.grid || {},
        stamps: new Uint32Array((layer.entities || []).length),
        fullPath: null,
        cellPath: null,
        cellKey: "",
        markers: (layer.entities || []).filter((entity) => entity.type === "point"),
      }));
    },
    resize() {
      const canvas = this.$refs.canvas;
      if (!canvas) return;
      const rect = this.$el.getBoundingClientRect();
      const ratio = window.devicePixelRatio || 1;
      canvas.width = Math.max(1, Math.round(rect.width * ratio));
      canvas.height = Math.max(1, Math.round(rect.height * ratio));
      this.ratio = ratio;
      this.cssWidth = rect.width;
      this.cssHeight = rect.height;
      if (!this.view.fitted || !this.view.userMoved) this.fit();
      this.requestDraw();
    },
    fit() {
      if (!this.cssWidth || !this.cssHeight) return;
      const b = this.backgroundBounds || this.imageBounds || [0, 0, 1, 1];
      const spanX = Math.max(1e-3, b[2] - b[0]);
      const spanY = Math.max(1e-3, b[3] - b[1]);
      const k = 0.94 * Math.min(this.cssWidth / spanX, this.cssHeight / spanY);
      this.view.k = k;
      this.view.ox = (this.cssWidth - spanX * k) / 2 - b[0] * k;
      this.view.oy = (this.cssHeight + spanY * k) / 2 + b[1] * k;
      this.view.fitted = true;
      this.view.userMoved = false;
      this.requestDraw();
    },
    toWorld(x, y) {
      return [(x - this.view.ox) / this.view.k, (this.view.oy - y) / this.view.k];
    },
    toScreen(wx, wy) {
      return [this.view.ox + wx * this.view.k, this.view.oy - wy * this.view.k];
    },
    eventPoint(event) {
      const rect = this.$refs.canvas.getBoundingClientRect();
      return [event.clientX - rect.left, event.clientY - rect.top];
    },
    viewport() {
      const [x0, y1] = this.toWorld(0, 0);
      const [x1, y0] = this.toWorld(this.cssWidth || 0, this.cssHeight || 0);
      return [x0, y0, x1, y1];
    },
    appendEntity(path, entity) {
      const type = entity.type;
      if (type === "line") {
        path.moveTo(entity.start[0], entity.start[1]);
        path.lineTo(entity.end[0], entity.end[1]);
      } else if (type === "polyline") {
        const rings = [entity.points || [], ...(entity.holes || [])];
        rings.forEach((ring, index) => {
          if (ring.length < 2) return;
          path.moveTo(ring[0][0], ring[0][1]);
          for (let i = 1; i < ring.length; i += 1) path.lineTo(ring[i][0], ring[i][1]);
          if (index > 0 || entity.closed) path.closePath();
        });
      } else if (type === "circle") {
        const [cx, cy] = entity.center;
        path.moveTo(cx + entity.radius, cy);
        path.arc(cx, cy, entity.radius, 0, Math.PI * 2);
      } else if (type === "arc") {
        const [cx, cy] = entity.center;
        const start = (Number(entity.start_angle) * Math.PI) / 180;
        const end = (Number(entity.end_angle) * Math.PI) / 180;
        path.moveTo(cx + entity.radius * Math.cos(start), cy + entity.radius * Math.sin(start));
        path.arc(cx, cy, entity.radius, start, end, false);
      } else if (type === "insert") {
        const [px, py] = entity.point;
        const angle = (Number(entity.rotation || 0) * Math.PI) / 180;
        const sx = Number(entity.xscale || 1);
        const sy = Number(entity.yscale || 1);
        const cos = Math.cos(angle);
        const sin = Math.sin(angle);
        const corners = [[0, 0], [1, 0], [1, 1], [0, 1]].map(([u, v]) => [px + u * sx * cos - v * sy * sin, py + u * sx * sin + v * sy * cos]);
        path.moveTo(corners[0][0], corners[0][1]);
        for (let i = 1; i < 4; i += 1) path.lineTo(corners[i][0], corners[i][1]);
        path.closePath();
      }
    },
    layerPath(layer, viewport) {
      if (!layer.bounds || !layer.cellSize) return null;
      const [left, bottom, right, top] = layer.bounds;
      const [vx0, vy0, vx1, vy1] = viewport;
      if (vx1 < left || vx0 > right || vy1 < bottom || vy0 > top) return null;
      const size = layer.cellSize;
      const lastCellX = Math.floor((right - left) / size);
      const lastCellY = Math.floor((top - bottom) / size);
      const firstX = Math.max(0, Math.floor((vx0 - left) / size));
      const lastX = Math.min(lastCellX, Math.floor((vx1 - left) / size));
      const firstY = Math.max(0, Math.floor((vy0 - bottom) / size));
      const lastY = Math.min(lastCellY, Math.floor((vy1 - bottom) / size));
      const visibleCells = (lastX - firstX + 1) * (lastY - firstY + 1);
      const totalCells = (lastCellX + 1) * (lastCellY + 1);
      if (visibleCells >= totalCells * 0.6) {
        if (!layer.fullPath) {
          layer.fullPath = new Path2D();
          for (const entity of layer.entities) this.appendEntity(layer.fullPath, entity);
        }
        return layer.fullPath;
      }
      const key = `${firstX}:${lastX}:${firstY}:${lastY}`;
      if (layer.cellKey === key && layer.cellPath) return layer.cellPath;
      this.stampFrame += 1;
      const stamp = this.stampFrame;
      const path = new Path2D();
      for (let x = firstX; x <= lastX; x += 1) {
        for (let y = firstY; y <= lastY; y += 1) {
          const indices = layer.grid[`${x}:${y}`];
          if (!indices) continue;
          for (const index of indices) {
            if (layer.stamps[index] === stamp) continue;
            layer.stamps[index] = stamp;
            this.appendEntity(path, layer.entities[index]);
          }
        }
      }
      layer.cellKey = key;
      layer.cellPath = path;
      return path;
    },
    requestDraw() {
      if (this.frameRequested) return;
      this.frameRequested = true;
      window.requestAnimationFrame(() => {
        this.frameRequested = false;
        this.draw();
      });
    },
    draw() {
      const canvas = this.$refs.canvas;
      if (!canvas) return;
      const ctx = canvas.getContext("2d");
      const ratio = this.ratio || 1;
      const bg = this.background && this.background.length === 3 ? this.background : [30, 34, 38];
      ctx.setTransform(1, 0, 0, 1, 0, 0);
      ctx.globalAlpha = 1;
      ctx.fillStyle = `rgb(${bg[0]},${bg[1]},${bg[2]})`;
      ctx.fillRect(0, 0, canvas.width, canvas.height);
      ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
      const imageOpacity = Math.max(0, Math.min(1, Number(this.imageOpacity ?? 1)));
      const dxfOpacity = Math.max(0, Math.min(1, Number(this.dxfOpacity ?? 1)));
      if (this.imageUrl && this.img.complete && this.img.naturalWidth && imageOpacity > 0) {
        const b = this.imageBounds || this.backgroundBounds;
        const [x0, y0] = this.toScreen(b[0], b[3]);
        const [x1, y1] = this.toScreen(b[2], b[1]);
        ctx.globalAlpha = imageOpacity;
        ctx.imageSmoothingEnabled = (x1 - x0) < this.img.naturalWidth * 2;
        ctx.drawImage(this.img, x0, y0, x1 - x0, y1 - y0);
      }
      if (!this.hideDxf && dxfOpacity > 0 && this.layers.length) {
        const hidden = new Set(this.hiddenLayers || []);
        const viewport = this.viewport();
        ctx.globalAlpha = dxfOpacity;
        for (const layer of this.layers) {
          if (hidden.has(layer.name)) continue;
          const path = this.layerPath(layer, viewport);
          if (path) {
            ctx.save();
            ctx.setTransform(ratio * this.view.k, 0, 0, -ratio * this.view.k, ratio * this.view.ox, ratio * this.view.oy);
            ctx.lineWidth = 1.25 / this.view.k;
            ctx.strokeStyle = layer.color;
            ctx.stroke(path);
            ctx.restore();
          }
          if (layer.markers.length) {
            ctx.strokeStyle = layer.color;
            ctx.lineWidth = 1.25;
            ctx.beginPath();
            for (const entity of layer.markers) {
              const [sx, sy] = this.toScreen(entity.point[0], entity.point[1]);
              if (sx < -5 || sy < -5 || sx > this.cssWidth + 5 || sy > this.cssHeight + 5) continue;
              ctx.moveTo(sx - 4, sy);
              ctx.lineTo(sx + 4, sy);
              ctx.moveTo(sx, sy - 4);
              ctx.lineTo(sx, sy + 4);
            }
            ctx.stroke();
          }
        }
      }
      ctx.globalAlpha = 1;
      if (this.measurePoints.length) {
        ctx.strokeStyle = "#ffd23f";
        ctx.fillStyle = "#ffd23f";
        ctx.lineWidth = 2;
        ctx.font = "12px sans-serif";
        const screen = this.measurePoints.map(([x, y]) => this.toScreen(x, y));
        ctx.beginPath();
        screen.forEach(([x, y], index) => (index ? ctx.lineTo(x, y) : ctx.moveTo(x, y)));
        ctx.stroke();
        for (const [x, y] of screen) {
          ctx.beginPath();
          ctx.arc(x, y, 4, 0, Math.PI * 2);
          ctx.fill();
        }
        if (this.measurePoints.length === 2) {
          const [p1, p2] = this.measurePoints;
          const distance = Math.hypot(p2[0] - p1[0], p2[1] - p1[1]);
          ctx.fillText(`${distance.toFixed(3)} m`, (screen[0][0] + screen[1][0]) / 2 + 6, (screen[0][1] + screen[1][1]) / 2 - 6);
        }
      }
    },
    onDown(event) {
      if (event.button !== 0) return;
      this.$refs.canvas.focus();
      const [x, y] = this.eventPoint(event);
      if (this.tool === "measure") {
        if (this.measurePoints.length >= 2) this.measurePoints = [];
        this.measurePoints.push(this.toWorld(x, y));
        if (this.measurePoints.length === 2) {
          const [p1, p2] = this.measurePoints;
          this.$emit("measure", { p1, p2, distance: Math.hypot(p2[0] - p1[0], p2[1] - p1[1]) });
        }
        this.requestDraw();
        return;
      }
      this.$refs.canvas.setPointerCapture(event.pointerId);
      this.drag = { x, y, ox: this.view.ox, oy: this.view.oy };
      this.view.userMoved = true;
      this.$refs.canvas.style.cursor = "grabbing";
    },
    onMove(event) {
      const [x, y] = this.eventPoint(event);
      const [wx, wy] = this.toWorld(x, y);
      this.$refs.status.textContent = `X ${wx.toFixed(3)} m · Y ${wy.toFixed(3)} m`;
      if (this.drag) {
        this.view.ox = this.drag.ox + (x - this.drag.x);
        this.view.oy = this.drag.oy + (y - this.drag.y);
        this.requestDraw();
      }
    },
    onUp(event) {
      if (this.drag && this.$refs.canvas.hasPointerCapture(event.pointerId)) {
        this.$refs.canvas.releasePointerCapture(event.pointerId);
      }
      this.drag = null;
      this.$refs.canvas.style.cursor = this.tool === "measure" ? "crosshair" : "grab";
    },
    onLeave() {
      if (!this.drag) this.$refs.status.textContent = "";
    },
    onWheel(event) {
      const [x, y] = this.eventPoint(event);
      const factor = Math.exp(-event.deltaY * 0.0015);
      const k = Math.min(1e6, Math.max(1e-5, this.view.k * factor));
      const [wx, wy] = this.toWorld(x, y);
      this.view.userMoved = true;
      this.view.k = k;
      this.view.ox = x - wx * k;
      this.view.oy = y + wy * k;
      this.requestDraw();
    },
  },
};
