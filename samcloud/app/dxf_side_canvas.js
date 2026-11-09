// Side view canvas for the SAMcloud DXF page: level bands, draggable floor/ceiling lines, add level, measure.
export default {
  template: `
    <div class="sam-dxf-side" style="position:relative;width:100%;height:100%;min-height:360px;overflow:hidden;border:1px solid #28475b;border-radius:9px;background:#09131d">
      <canvas ref="canvas" style="display:block;width:100%;height:100%;touch-action:none"
        @pointerdown="onDown" @pointermove="onMove" @pointerup="onUp" @pointerleave="onLeave" @wheel.prevent="onWheel"></canvas>
      <div ref="status" style="position:absolute;left:10px;bottom:8px;padding:4px 8px;border-radius:5px;background:rgba(9,19,29,.86);color:#b9ccda;font-size:12px;pointer-events:none"></div>
      <div v-if="!image" style="position:absolute;inset:0;display:grid;place-items:center;color:#96aabe;font-size:14px;pointer-events:none">Press "Render side view" to draw the preview cloud.</div>
    </div>
  `,
  props: {
    image: String,
    bounds: Array,
    levels: Array,
    tool: String,
    axisLabel: String,
  },
  data() {
    return { localLevels: [], measurePoints: [] };
  },
  mounted() {
    this.view = { k: 1, ox: 0, oy: 0, fitted: false };
    this.drag = null;
    this.img = new Image();
    this.img.onload = () => {
      if (!this.view.fitted) this.fit();
      this.draw();
    };
    this.localLevels = this.copyLevels(this.levels);
    this.observer = new ResizeObserver(() => this.resize());
    this.observer.observe(this.$el);
    this.resize();
    if (this.image) this.img.src = this.image;
  },
  unmounted() {
    if (this.observer) this.observer.disconnect();
  },
  watch: {
    image(value) {
      this.view.fitted = false;
      this.measurePoints = [];
      if (value) this.img.src = value;
      this.draw();
    },
    bounds() {
      this.view.fitted = false;
      this.fit();
    },
    levels: {
      handler(value) {
        if (!this.drag || this.drag.kind !== "line") this.localLevels = this.copyLevels(value);
        this.draw();
      },
      deep: true,
    },
    tool() {
      this.measurePoints = [];
      this.updateCursor();
      this.draw();
    },
  },
  methods: {
    copyLevels(levels) {
      return (levels || []).map((level) => ({ ...level }));
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
      this.draw();
    },
    worldBounds() {
      const b = this.bounds && this.bounds.length === 4 ? this.bounds : [0, 1, 0, 1];
      return { hMin: b[0], hMax: b[1], zMin: b[2], zMax: b[3] };
    },
    fit() {
      if (!this.cssWidth || !this.cssHeight) return;
      const b = this.worldBounds();
      const spanH = Math.max(1e-3, b.hMax - b.hMin);
      const spanZ = Math.max(1e-3, b.zMax - b.zMin);
      const k = 0.92 * Math.min(this.cssWidth / spanH, this.cssHeight / spanZ);
      this.view.k = k;
      this.view.ox = (this.cssWidth - spanH * k) / 2 - b.hMin * k;
      this.view.oy = (this.cssHeight + spanZ * k) / 2 + b.zMin * k;
      this.view.fitted = true;
      this.view.userMoved = false;
      this.draw();
    },
    toScreen(h, z) {
      return [this.view.ox + h * this.view.k, this.view.oy - z * this.view.k];
    },
    toWorld(x, y) {
      return [(x - this.view.ox) / this.view.k, (this.view.oy - y) / this.view.k];
    },
    eventPoint(event) {
      const rect = this.$refs.canvas.getBoundingClientRect();
      return [event.clientX - rect.left, event.clientY - rect.top];
    },
    hitLine(y) {
      let best = null;
      for (const level of this.localLevels) {
        for (const edge of ["floor", "ceiling"]) {
          const z = edge === "floor" ? level.floor_z : level.ceiling_z;
          if (z === null || z === undefined) continue;
          const distance = Math.abs(this.toScreen(0, z)[1] - y);
          if (distance <= 6 && (!best || distance < best.distance)) best = { id: level.id, edge, distance };
        }
      }
      return best;
    },
    updateCursor(y) {
      const canvas = this.$refs.canvas;
      if (!canvas) return;
      if (this.tool === "add" || this.tool === "measure") canvas.style.cursor = "crosshair";
      else if (this.drag && this.drag.kind === "line") canvas.style.cursor = "ns-resize";
      else if (this.drag) canvas.style.cursor = "grabbing";
      else if (y !== undefined && this.hitLine(y)) canvas.style.cursor = "ns-resize";
      else canvas.style.cursor = "grab";
    },
    onDown(event) {
      const [x, y] = this.eventPoint(event);
      const [h, z] = this.toWorld(x, y);
      if (event.button !== 0) return;
      if (this.tool === "add") {
        this.$emit("add_level", { z });
        return;
      }
      if (this.tool === "measure") {
        if (this.measurePoints.length >= 2) this.measurePoints = [];
        this.measurePoints.push([h, z]);
        if (this.measurePoints.length === 2) {
          const [p1, p2] = this.measurePoints;
          const distance = Math.hypot(p2[0] - p1[0], p2[1] - p1[1]);
          this.$emit("measure", { p1, p2, distance });
        }
        this.draw();
        return;
      }
      const hit = this.hitLine(y);
      this.$refs.canvas.setPointerCapture(event.pointerId);
      if (hit) this.drag = { kind: "line", id: hit.id, edge: hit.edge };
      else this.drag = { kind: "pan", x, y, ox: this.view.ox, oy: this.view.oy };
      this.view.userMoved = true;
      this.updateCursor(y);
    },
    onMove(event) {
      const [x, y] = this.eventPoint(event);
      const [h, z] = this.toWorld(x, y);
      this.$refs.status.textContent = `${this.axisLabel || "H"} ${h.toFixed(3)} m · Z ${z.toFixed(3)} m`;
      if (this.drag && this.drag.kind === "pan") {
        this.view.ox = this.drag.ox + (x - this.drag.x);
        this.view.oy = this.drag.oy + (y - this.drag.y);
        this.draw();
      } else if (this.drag && this.drag.kind === "line") {
        const level = this.localLevels.find((item) => item.id === this.drag.id);
        if (level) {
          if (this.drag.edge === "floor") level.floor_z = z;
          else level.ceiling_z = z;
          this.drag.z = z;
          this.draw();
        }
      } else {
        this.updateCursor(y);
      }
    },
    onUp(event) {
      if (this.drag && this.drag.kind === "line" && this.drag.z !== undefined) {
        this.$emit("level_move", { id: this.drag.id, edge: this.drag.edge, z: this.drag.z });
      }
      if (this.drag && this.$refs.canvas.hasPointerCapture(event.pointerId)) {
        this.$refs.canvas.releasePointerCapture(event.pointerId);
      }
      this.drag = null;
      this.updateCursor(this.eventPoint(event)[1]);
    },
    onLeave() {
      if (!this.drag) this.$refs.status.textContent = "";
    },
    onWheel(event) {
      const [x, y] = this.eventPoint(event);
      const factor = Math.exp(-event.deltaY * 0.0015);
      const k = Math.min(1e6, Math.max(1e-4, this.view.k * factor));
      const [h, z] = this.toWorld(x, y);
      this.view.userMoved = true;
      this.view.k = k;
      this.view.ox = x - h * k;
      this.view.oy = y + z * k;
      this.draw();
    },
    draw() {
      const canvas = this.$refs.canvas;
      if (!canvas) return;
      const ctx = canvas.getContext("2d");
      ctx.setTransform(1, 0, 0, 1, 0, 0);
      ctx.fillStyle = "#09131d";
      ctx.fillRect(0, 0, canvas.width, canvas.height);
      ctx.setTransform(this.ratio || 1, 0, 0, this.ratio || 1, 0, 0);
      const b = this.worldBounds();
      if (this.image && this.img.complete && this.img.naturalWidth) {
        const [x0, y0] = this.toScreen(b.hMin, b.zMax);
        const [x1, y1] = this.toScreen(b.hMax, b.zMin);
        ctx.imageSmoothingEnabled = this.view.k < 200;
        ctx.drawImage(this.img, x0, y0, x1 - x0, y1 - y0);
      }
      const width = this.cssWidth || 0;
      ctx.font = "12px sans-serif";
      for (const level of this.localLevels) {
        const colour = level.checked ? "38,183,239" : "150,170,190";
        const yFloor = this.toScreen(0, level.floor_z)[1];
        const hasCeiling = level.ceiling_z !== null && level.ceiling_z !== undefined;
        const yCeiling = hasCeiling ? this.toScreen(0, level.ceiling_z)[1] : null;
        if (hasCeiling) {
          ctx.fillStyle = `rgba(${colour},0.12)`;
          ctx.fillRect(0, Math.min(yFloor, yCeiling), width, Math.abs(yFloor - yCeiling));
        }
        ctx.strokeStyle = `rgba(${colour},0.95)`;
        ctx.lineWidth = 2;
        ctx.setLineDash([]);
        ctx.beginPath();
        ctx.moveTo(0, yFloor);
        ctx.lineTo(width, yFloor);
        ctx.stroke();
        if (hasCeiling) {
          ctx.setLineDash([8, 5]);
          ctx.beginPath();
          ctx.moveTo(0, yCeiling);
          ctx.lineTo(width, yCeiling);
          ctx.stroke();
          ctx.setLineDash([]);
        }
        ctx.fillStyle = `rgba(${colour},1)`;
        ctx.fillText(`Level ${level.name} · floor ${Number(level.floor_z).toFixed(2)} m`, 8, yFloor - 5);
        if (hasCeiling) ctx.fillText(`ceiling ${Number(level.ceiling_z).toFixed(2)} m`, 8, yCeiling + 14);
      }
      if (this.measurePoints.length) {
        ctx.strokeStyle = "#ffd23f";
        ctx.fillStyle = "#ffd23f";
        ctx.lineWidth = 2;
        const screen = this.measurePoints.map(([h, z]) => this.toScreen(h, z));
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
  },
};
